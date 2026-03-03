"""
recsys_api.py — PF Recommendation API 
"""

import os
import json
import logging
import hashlib
import time
import math
import numpy as np
import redis.asyncio as redis
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Header
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch

load_dotenv()
logger = logging.getLogger("recsys_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
RANKING_API_URL = os.getenv("RANKING_API_URL", "http://ranking-api:8002")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "pf-inventory-v1")

MAX_PER_AGENT, MAX_PER_LOCATION = 2, 3
KNN_FETCH_SIZE = 60
DEFAULT_LIMIT, MAX_LIMIT = 8, 20
BLEND_ANCHOR, BLEND_USER = 0.65, 0.35
CACHE_TTL_ANON, CACHE_TTL_PERSONALISED = 300, 120

_RANKING_FIELDS = {
    'property_listing_id', 'price', 'price_period', 'category_id',
    'property_type_id', 'bedrooms', 'bathrooms', 'size_sqft',
    'listing_level', 'quality_score', 'super_agent_score',
    'popularity_score', 'listing_date',
}

# Added location_coordinates so we can extract them for Distance calculation
_FETCH_FIELDS = list(set(_RANKING_FIELDS).union({"agent_id", "location_id", "location_coordinates"}))


os_client = AsyncOpenSearch(
    hosts=[{'host': OS_HOST, 'port': int(os.getenv("OPENSEARCH_PORT", 9200))}],
    use_ssl=os.getenv("OPENSEARCH_SSL", "false").lower() == "true",
    verify_certs=os.getenv("OPENSEARCH_SSL", "false").lower() == "true",
    timeout=30
)
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
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

app = FastAPI(title="PF-RecSys-API", version="15.2-GeoAware", lifespan=lifespan)

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

def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

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
        return coords[1], coords[0]  # OS geo_point arrays are typically [lon, lat]
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        R = 6371.0 # Radius of earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return 0.0

async def get_rank_scores(properties_list: list) -> dict:
    if not properties_list or _http_client is None:
        return {}

    if await _ranking_breaker.is_open():
        logger.warning("Ranking circuit breaker is OPEN — skipping ranking, using quality_score fallback")
        return {}

    try:
        resp = await _http_client.post(
            f"{RANKING_API_URL}/rank",
            json={"properties": properties_list}
        )
        if resp.status_code == 200:
            await _ranking_breaker.record_success()
            return {
                str(r['property_listing_id']): r['rank_score']
                for r in resp.json().get("ranked_results", [])
            }
        else:
            await _ranking_breaker.record_failure()
            logger.warning(f"Ranking API returned {resp.status_code}")
    except httpx.TimeoutException:
        await _ranking_breaker.record_failure()
        logger.warning("Ranking API timeout (>2s) — falling back to quality_score")
    except Exception as e:
        await _ranking_breaker.record_failure()
        logger.warning(f"Ranking API error: {type(e).__name__}: {e}")
    return {}

async def get_user_dna(user_id: str) -> Optional[dict]:
    if not user_id:
        return None
    try:
        raw = await redis_client.get(f"user_dna:{user_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None

def compute_search_vector(anchor_vec: np.ndarray, user_id: str, dna: Optional[dict]) -> np.ndarray:
    if not dna:
        return anchor_vec

    v1 = np.array(dna.get('vector_1', []))
    v2 = np.array(dna.get('vector_2', []))

    if v1.shape != anchor_vec.shape:
        return anchor_vec

    sim1 = cosine_similarity(anchor_vec, v1)
    sim2 = cosine_similarity(anchor_vec, v2) if v2.shape == anchor_vec.shape else -1
    best_user_vec = v1 if sim1 >= sim2 else v2

    blended = (BLEND_ANCHOR * anchor_vec) + (BLEND_USER * best_user_vec)
    norm = np.linalg.norm(blended)
    return blended / norm if norm > 0 else blended

def apply_diversity_filter(hits: list, limit: int) -> list:
    agent_counts, location_counts, filtered = {}, {}, []
    for row in hits:
        agent_id = str(row.get('agent_id', 'unknown'))
        location_id = str(row.get('location_id', 'unknown'))
        if agent_counts.get(agent_id, 0) >= MAX_PER_AGENT:
            continue
        if location_counts.get(location_id, 0) >= MAX_PER_LOCATION:
            continue

        agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        location_counts[location_id] = location_counts.get(location_id, 0) + 1
        filtered.append(row)
        if len(filtered) >= limit:
            break
    return filtered

async def _cache_get(key: str) -> Optional[list]:
    try:
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None

async def _cache_set(key: str, data: list, ttl: int) -> None:
    try:
        await redis_client.set(key, json.dumps(data), ex=ttl)
    except Exception:
        pass

@app.get("/internal/v1/recommendations/{id}")
async def get_recommendations(
    id: str,
    locale: str = Query("en"),
    filters_price_type: Optional[str] = Query(None, alias="filters.price_type"),
    x_pf_user_id: Optional[str] = Header(None, alias="X-PF-User-Id"),
    limit: int = Query(DEFAULT_LIMIT, le=MAX_LIMIT),
):
    user_part = hashlib.sha256(x_pf_user_id.encode()).hexdigest()[:16] if x_pf_user_id else "anon"
    cache_key = f"recsys:{id}:{locale}:{filters_price_type or 'any'}:{user_part}"

    cached = await _cache_get(cache_key)
    if cached is not None:
        return {"properties": cached[:limit]}

    anchor = None
    try:
        # STEP 1: Try direct GET by ID
        anchor_resp = await os_client.get(
            index=INDEX_NAME, id=id,
            _source_includes=["property_vector", "category_id", "property_type_id", "property_listing_id", "location_coordinates"]
        )
        anchor = anchor_resp['_source']
    except Exception:
        # STEP 2: Try flexible match
        try:
            fallback_resp = await os_client.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"property_listing_id": id}},
                                {"term": {"property_listing_id": int(id) if id.isdigit() else 0}}
                            ]
                        }
                    },
                    "_source": ["property_vector", "category_id", "property_type_id", "property_listing_id", "location_coordinates"]
                }
            )
            hits = fallback_resp.get('hits', {}).get('hits', [])
            if hits:
                anchor = hits[0]['_source']
                logger.info(f"Found anchor listing {id} via fallback search.")
        except Exception as search_err:
            logger.warning(f"Search for ID {id} failed: {search_err}")

    if not anchor:
        logger.warning(f"Listing {id} is NOT in the current OpenSearch index.")
        return JSONResponse(
            status_code=404, 
            content={"properties": [], "error": f"ID {id} not found."}
        )

    anchor_vec = anchor.get('property_vector')
    if not anchor_vec:
        logger.warning(f"Anchor listing {id} has no property_vector")
        return {"properties": []}
    
    anchor_vec = np.array(anchor_vec)
    if anchor_vec.ndim != 1 or anchor_vec.shape[0] == 0:
        logger.warning(f"Anchor listing {id} has invalid vector shape: {anchor_vec.shape}")
        return {"properties": []}

    dna = await get_user_dna(x_pf_user_id) if x_pf_user_id else None
    search_vec = compute_search_vector(anchor_vec, x_pf_user_id, dna)

    try:
        cat_id = int(anchor.get('category_id', 0))
    except (ValueError, TypeError):
        cat_id = anchor.get('category_id')

    try:
        prop_type_id = int(anchor.get('property_type_id', 0))
    except (ValueError, TypeError):
        prop_type_id = anchor.get('property_type_id')

    knn_must = [
        {"term": {"category_id": cat_id}},
        {"term": {"property_type_id": prop_type_id}}
    ]
    
    if filters_price_type:
        knn_must.append({"term": {"price_period": filters_price_type}})

    doc_id = anchor.get('property_listing_id', id)

    knn_query = {
        "size": KNN_FETCH_SIZE,
        "_source": _FETCH_FIELDS,
        "query": {
            "knn": {
                "property_vector": {
                    "vector": search_vec.tolist(),
                    "k": KNN_FETCH_SIZE,
                    "filter": {
                        "bool": {
                            "must": knn_must,
                            "must_not": [{"term": {"property_listing_id": doc_id}}]
                        }
                    }
                }
            }
        },
    }

    try:
        resp = await os_client.search(index=INDEX_NAME, body=knn_query)
        hits = [h['_source'] for h in resp['hits']['hits']]
        
        if not hits:
            logger.warning(f"Zero KNN results found for ID {id}")
            return {"properties": []}
    except Exception as e:
        logger.error(f"OpenSearch KNN search failed for ID {id}: {e}")
        return {"properties": []}

    # Extract Anchor Lat/Lon
    anchor_lat, anchor_lon = _extract_lat_lon(anchor.get('location_coordinates'))

    # Build the payload and append the Haversine distance
    ranking_payload = []
    for h in hits:
        payload_item = {k: v for k, v in h.items() if k in _RANKING_FIELDS}
        
        h_lat, h_lon = _extract_lat_lon(h.get('location_coordinates'))
        payload_item['distance_km'] = haversine(anchor_lat, anchor_lon, h_lat, h_lon)
        
        ranking_payload.append(payload_item)

    score_map = await get_rank_scores(ranking_payload)

    if score_map:
        hits.sort(key=lambda x: score_map.get(str(x.get('property_listing_id')), 0), reverse=True)
    else:
        hits.sort(key=lambda x: float(x.get('quality_score', 0)), reverse=True)

    diverse_hits = apply_diversity_filter(hits, MAX_LIMIT)

    properties = [
        {
            "id": str(row.get('property_listing_id')),
            "rank_score": round(score_map.get(str(row.get('property_listing_id')), 0.0), 6),
        }
        for row in diverse_hits
    ]

    await _cache_set(cache_key, properties, CACHE_TTL_PERSONALISED if dna else CACHE_TTL_ANON)
    return {"properties": properties[:limit]}