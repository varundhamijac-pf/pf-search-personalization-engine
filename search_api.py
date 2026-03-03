"""
search_api.py — PF Search API (v15.2-GeoAware)
"""

import os
import json
import hashlib
import logging
import time
import math
import httpx
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch

from pf_features import AMENITY_MAP

load_dotenv()
logger = logging.getLogger("search_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
RANKING_API_URL = os.getenv("RANKING_API_URL", "http://ranking-api:8002")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "pf-inventory-v1")

CACHE_TTL_SECONDS = 60
CACHE_TTL_EMPTY_SECONDS = 10
MAX_OS_RESULT_WINDOW = 10000

_RANKING_FIELDS = {
    'property_listing_id', 'price', 'price_period', 'category_id',
    'property_type_id', 'bedrooms', 'bathrooms', 'size_sqft',
    'listing_level', 'quality_score', 'super_agent_score',
    'popularity_score', 'listing_date',
}

# FIX: Safely bind to 9200 without SSL for local Docker communication
os_client = AsyncOpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(os.getenv("OPENSEARCH_PORT", 9200))}],
    use_ssl=os.getenv("OPENSEARCH_SSL", "false").lower() == "true",
    verify_certs=os.getenv("OPENSEARCH_SSL", "false").lower() == "true"
)
redis_client = aioredis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

_http_client: Optional[httpx.AsyncClient] = None

class RedisCircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self.threshold = threshold
        self.cooldown = int(cooldown)
        self.fail_key = "circuit:ranking:fails"
        self.open_key = "circuit:ranking:open"

    async def is_open(self) -> bool:
        return await redis_client.exists(self.open_key)

    async def record_success(self):
        await redis_client.delete(self.fail_key)

    async def record_failure(self):
        fails = await redis_client.incr(self.fail_key)
        if fails == 1:
            await redis_client.expire(self.fail_key, self.cooldown)
        if fails >= self.threshold:
            await redis_client.set(self.open_key, "1", ex=self.cooldown)
            await redis_client.delete(self.fail_key)
            logger.error(f"Circuit breaker OPEN globally for {self.cooldown}s.")

_ranking_breaker = RedisCircuitBreaker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        timeout=2.0 
    )
    yield
    await _http_client.aclose()

app = FastAPI(title="PF-Search-API", version="15.2-GeoAware", lifespan=lifespan)

@app.get('/health')
async def health():
    try:
        await redis_client.ping()
        await os_client.cluster.health()
        return {
            'status': 'ok',
            'ranking_circuit': 'open' if await _ranking_breaker.is_open() else 'closed',
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={'status': 'degraded', 'error': str(e)})

def _extract_lat_lon(coords):
    """Safely extract lat/lon from various OpenSearch geometry formats"""
    if not coords:
        return None, None
    if isinstance(coords, dict):
        return coords.get('lat'), coords.get('lon')
    if isinstance(coords, str):
        try:
            lat, lon = coords.split(',')
            return float(lat.strip()), float(lon.strip())
        except ValueError:
            return None, None
    if isinstance(coords, list) and len(coords) == 2:
        return coords[1], coords[0]  
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return 0.0

class SearchFilters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    category_id: Optional[int] = Field(None, alias="offering_type")
    property_type_ids: Optional[List[int]] = []
    bedrooms: Optional[List[int]] = []
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    price_period: Optional[str] = None
    amenities: Optional[List[str]] = []
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = 5.0
    completion_status: Optional[str] = None
    furnished: Optional[str] = None
    is_verified: Optional[bool] = None

class SearchRequest(BaseModel):
    user_id: Optional[str] = None
    filters: SearchFilters
    limit: int = 50
    page: int = 1
    sort: str = "mlWeights"

OPENSEARCH_SORT_MAP = {
    "newest":       [{"listing_date": {"order": "desc"}}],
    "priceAsc":     [{"price": {"order": "asc"}}],
    "priceDesc":    [{"price": {"order": "desc"}}],
    "bedroomAsc":   [{"bedrooms": {"order": "asc"}}, {"price": {"order": "asc"}}],
    "bedroomDesc":  [{"bedrooms": {"order": "desc"}}, {"price": {"order": "asc"}}],
    "freshnessDate": [{"listing_date": {"order": "desc"}}],
}

ML_RANKED_SORTS = {
    "featured", "mlWeights",
    "caratRankingVariantA", "caratRankingVariantB",
    "freshnessVariantA",
    "superAgentVariantA", "superAgentVariantB", "superAgentVariantC",
    "exclusiveBoost", "doubleExclusiveBoost", "repeatOffender",
}

def _search_cache_key(locale: str, req: SearchRequest) -> str:
    payload = {
        "locale": locale, "filters": req.filters.model_dump(),
        "limit": req.limit, "page": req.page, "sort": req.sort,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    return f"serp:{fingerprint}"

async def _cache_get(key: str) -> Optional[dict]:
    try:
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None

async def _cache_set(key: str, data: dict, ttl: int = CACHE_TTL_SECONDS) -> None:
    try:
        await redis_client.set(key, json.dumps(data), ex=ttl)
    except Exception:
        pass

async def get_rank_scores(properties_list: list) -> list:
    if not properties_list or _http_client is None:
        return []

    if await _ranking_breaker.is_open():
        logger.warning("Ranking circuit breaker OPEN — skipping ML ranking")
        return []

    try:
        resp = await _http_client.post(
            f"{RANKING_API_URL}/rank",
            json={"properties": properties_list}
        )
        if resp.status_code == 200:
            await _ranking_breaker.record_success()
            return resp.json().get("ranked_results", [])
        else:
            await _ranking_breaker.record_failure()
            logger.warning(f"Ranking API returned {resp.status_code}")
    except httpx.TimeoutException:
        await _ranking_breaker.record_failure()
        logger.warning("Ranking API timeout (>2s) — falling back to quality_score")
    except Exception as e:
        await _ranking_breaker.record_failure()
        logger.warning(f"Ranking API error: {type(e).__name__}: {e}")
    return []

def build_query(f: SearchFilters, limit: int, offset: int = 0,
                drop_geo: bool = False, sort_clauses: list = None) -> dict:
    must = []
    if f.category_id:
        must.append({"term": {"category_id": f.category_id}})
    if f.property_type_ids:
        must.append({"terms": {"property_type_id": f.property_type_ids}})
    if f.bedrooms:
        must.append({"terms": {"bedrooms": f.bedrooms}})
    if f.completion_status:
        must.append({"term": {"completion_status": f.completion_status}})
    if f.furnished:
        must.append({"term": {"furnished_flag": f.furnished}})
    if f.is_verified is not None:
        must.append({"term": {"is_verified": f.is_verified}})

    period = f.price_period or ("yearly" if f.category_id in [2, 4] else None)
    if period:
        must.append({"term": {"price_period": period}})

    if f.min_price is not None or f.max_price is not None:
        r = {}
        if f.min_price is not None:
            r["gte"] = f.min_price
        if f.max_price is not None:
            r["lte"] = f.max_price
        must.append({"range": {"price": r}})

    if f.amenities:
        ids = [AMENITY_MAP[a] for a in f.amenities if a in AMENITY_MAP]
        if ids:
            must.append({"terms": {"amenities": ids}})

    geo = []
    if not drop_geo and f.latitude is not None and f.longitude is not None:
        geo.append({
            "geo_distance": {
                "distance": f"{f.radius_km}km",
                "location_coordinates": {"lat": f.latitude, "lon": f.longitude}
            }
        })

    query = {
        "size": limit,
        "from": offset,
        "_source": {"excludes": ["property_vector"]},
        "query": {"bool": {"must": must, "filter": geo}},
    }

    if sort_clauses:
        query["sort"] = sort_clauses

    return query

async def _execute_search(locale: str, req: SearchRequest) -> dict:
    cache_key = _search_cache_key(locale, req)
    cached = await _cache_get(cache_key)
    if cached is not None:
        return cached

    offset = max(0, min((req.page - 1) * req.limit, MAX_OS_RESULT_WINDOW - req.limit))
    use_ml_ranking = req.sort in ML_RANKED_SORTS
    os_sort = OPENSEARCH_SORT_MAP.get(req.sort)

    fetch_multiplier = 3 if use_ml_ranking else 1

    fallback_tiers = [
        {"filters": req.filters, "label": "exact"},
        {"filters": req.filters.model_copy(deep=True), "label": "no_amenities"},
        {"filters": req.filters.model_copy(deep=True), "label": "relaxed_price"},
        {"filters": req.filters.model_copy(deep=True), "label": "relaxed_price_no_geo"},
    ]
    fallback_tiers[1]["filters"].amenities = []
    if fallback_tiers[2]["filters"].max_price:
        fallback_tiers[2]["filters"].max_price *= 1.2
    if fallback_tiers[3]["filters"].max_price:
        fallback_tiers[3]["filters"].max_price *= 1.2

    hits, total, applied_tier = [], 0, 0
    for tier_idx, tier in enumerate(fallback_tiers):
        try:
            body = build_query(
                tier["filters"],
                req.limit * fetch_multiplier,
                offset,
                drop_geo=(tier_idx == 3),
                sort_clauses=os_sort,
            )
            resp = await os_client.search(index=INDEX_NAME, body=body)
            hits = resp['hits']['hits']
            total = resp['hits']['total']['value']
            if total > 0:
                applied_tier = tier_idx
                if tier_idx > 0:
                    logger.info(f"Search fallback activated: tier={tier['label']} (tier {tier_idx})")
                break
        except Exception as e:
            logger.warning(f"Search tier {tier_idx} ({tier['label']}) failed: {e}")
            continue

    if not hits:
        result = {
            "meta": {"page": req.page, "total_count": 0, "per_page": req.limit, "page_count": 0},
            "properties": [],
            "fallback": False,
            "fallback_tier": None,
        }
        await _cache_set(cache_key, result, ttl=CACHE_TTL_EMPTY_SECONDS)
        return result

    sources = [h['_source'] for h in hits]

    score_map = {}
    if use_ml_ranking:
        ranking_payload = []
        search_lat = req.filters.latitude
        search_lon = req.filters.longitude

        for s in sources:
            payload_item = {k: v for k, v in s.items() if k in _RANKING_FIELDS}
            
            # Extract distance dynamically if the user provided a geo-pin!
            s_lat, s_lon = _extract_lat_lon(s.get('location_coordinates'))
            payload_item['distance_km'] = haversine(search_lat, search_lon, s_lat, s_lon)
            
            ranking_payload.append(payload_item)

        ranked = await get_rank_scores(ranking_payload)
        if ranked:
            score_map = {str(r['property_listing_id']): r['rank_score'] for r in ranked}
            sources.sort(
                key=lambda x: score_map.get(str(x.get('property_listing_id')), 0),
                reverse=True
            )
        else:
            sources.sort(key=lambda x: float(x.get('quality_score', 0)), reverse=True)
            score_map = {
                str(s.get('property_listing_id')): float(s.get('quality_score', 0))
                for s in sources
            }
            logger.info("Using quality_score fallback for search ranking")

    results = [
        {
            "id": str(row.get('property_listing_id')),
            "rank_score": round(score_map.get(str(row.get('property_listing_id')), 0.0), 6),
        }
        for row in sources[:req.limit]
    ]

    total_safe = min(total, MAX_OS_RESULT_WINDOW)

    result = {
        "meta": {
            "page": req.page,
            "total_count": total_safe,
            "per_page": req.limit,
            "page_count": max(1, (total_safe + req.limit - 1) // req.limit),
        },
        "properties": results,
        "fallback": (applied_tier > 0),
        "fallback_tier": fallback_tiers[applied_tier]["label"] if applied_tier > 0 else None,
    }

    await _cache_set(cache_key, result)
    return result

@app.post("/internal/v1/search/{locale}")
async def search_post(locale: str, req: SearchRequest):
    return await _execute_search(locale, req)

@app.get("/internal/v1/search/{locale}")
async def search_get(
    locale: str,
    category_id: Optional[int] = Query(None, alias="filters.category_id"),
    property_type_ids: Optional[str] = Query(None, alias="filters.property_type_ids"),
    bedrooms: Optional[str] = Query(None, alias="filters.bedrooms"),
    min_price: Optional[float] = Query(None, alias="filters.min_price"),
    max_price: Optional[float] = Query(None, alias="filters.max_price"),
    price_period: Optional[str] = Query(None, alias="filters.price_period"),
    amenities: Optional[str] = Query(None, alias="filters.amenities"),
    latitude: Optional[float] = Query(None, alias="filters.latitude"),
    longitude: Optional[float] = Query(None, alias="filters.longitude"),
    radius_km: Optional[float] = Query(5.0, alias="filters.radius_km"),
    completion_status: Optional[str] = Query(None, alias="filters.completion_status"),
    furnished: Optional[str] = Query(None, alias="filters.furnished"),
    is_verified: Optional[bool] = Query(None, alias="filters.is_verified"),
    limit: int = Query(50, alias="page.limit"),
    page: int = Query(1, alias="page.number"),
    sort: str = Query("mlWeights", alias="sorting.sort"),
):
    filters = SearchFilters(
        category_id=category_id,
        property_type_ids=[int(x) for x in property_type_ids.split(",")] if property_type_ids else [],
        bedrooms=[int(x) for x in bedrooms.split(",")] if bedrooms else [],
        min_price=min_price,
        max_price=max_price,
        price_period=price_period,
        amenities=amenities.split(",") if amenities else [],
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        completion_status=completion_status,
        furnished=furnished,
        is_verified=is_verified,
    )
    req = SearchRequest(filters=filters, limit=limit, page=page, sort=sort)
    return await _execute_search(locale, req)