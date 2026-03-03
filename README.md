# PropertyFinder Recommendation & Search System

A production ML pipeline for real estate recommendations and search ranking, operating across UAE, Egypt, Bahrain, and Saudi Arabia (~1.38M active listings).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        5-LANE PIPELINE                                  │
├─────────────┬───────────────────────────────────────────────────────────┤
│ LANE 1      │ Data & ETL         → redshift_data.py (AWS Lambda)       │
│ LANE 2      │ ML Training        → sagemaker_pipeline.py + pf_features │
│ LANE 3      │ Storage & Indexing → seed_data.py + OpenSearch + Redis   │
│ LANE 4      │ AI Serving         → search_api / recsys_api / ranking   │
│ LANE 5      │ Hydration          → BFF Gateway (returns thin IDs)      │
└─────────────┴───────────────────────────────────────────────────────────┘
```

**Key Design Principle — Thin Payload:** AI services return only `{ "id": "...", "rank_score": 0.95 }` (~200 bytes for 8 properties). The BFF hydrates full property details from SQL.

### Data Flow

- **Daily:** Redshift → Lambda ETL → S3 Parquet → SageMaker (vectors + User DNA + XGBRanker) → OpenSearch (blue-green swap) + Redis (DNA) + ranking_api (hot reload brain.pkl)
- **Hourly:** Delta ETL → new listing embeddings → OpenSearch delta upsert
- **Real-time:** User request → BFF → recsys_api (KNN + personalization + ranking) → thin IDs → BFF → SQL hydration → App

### Service Architecture

| Service | Port | Workers | Purpose |
|---------|------|---------|---------|
| `ranking_api` | 8002 | 1 | XGBoost inference, brain.pkl in memory |
| `recsys_api` | 8001 | 4 | KNN vector search + personalization + diversity |
| `search_api` | 8000 | 4 | SERP filters + 19 sort modes + 4-tier fallback |
| OpenSearch | 9200 | — | HNSW vector index (384-dim, cosine similarity) |
| Redis | 6379 | — | User DNA cache + response cache + circuit breaker |

## Project Files

```
├── pf_features.py              # Single source of truth: 13 MODEL_FEATURES, preprocess_for_model(), haversine
├── sagemaker_pipeline.py       # ML training: vectors → user DNA → XGBRanker → brain.pkl
├── redshift_data.py            # Lambda ETL: Redshift → S3 Parquet (inventory + interactions)
├── seed_data.py                # OpenSearch blue-green indexer + Redis DNA loader
├── ranking_api.py              # XGBoost inference microservice (POST /rank)
├── recsys_api.py               # Recommendations: KNN + personalization + geo-distance + diversity
├── search_api.py               # SERP: filters + fallback + ML ranking
├── opensearch_mapping.json     # Index schema with HNSW vector + geo_point config
├── docker-compose.yml          # Full stack: 3 APIs + OpenSearch + Redis
├── Dockerfile                  # Full ML stack (for SageMaker pipeline)
├── Dockerfile.api              # Lightweight API image (no sentence-transformers)
├── requirements.txt            # Full dependencies (training + serving)
├── requirements-api.txt        # API-only dependencies
├── test_distance_km.py         # Unit + integration tests for distance_km feature
├── test_recsys_quality.py      # 65+ quality checks: geo, diversity, filters, performance
├── test_diverse_properties.py  # 29-archetype recommendation quality test
├── test_api_scenarios.py       # Live API scenario tests across property types
└── artifacts/                  # Generated (not committed)
    ├── brain.pkl               # Trained XGBRanker model
    ├── inventory.parquet       # Listings with vectors
    └── user_vectors.parquet    # User DNA profiles
```

## ML Model

### XGBRanker — 13 Features

```python
MODEL_FEATURES = [
    'feature_annual_rent',   # Annualized rent (rent × multiplier)
    'feature_sale_price',    # Sale price (0 for rentals)
    'is_sale',               # Binary: sale (1) vs rent (0)
    'price_per_sqft',        # Price efficiency signal
    'listing_level_score',   # Premium=2, Featured=1, Standard=0
    'super_agent_score',     # Agent quality metric
    'popularity_score',      # 90-day time-decayed engagement score
    'bedrooms',              # Bedroom count
    'bathrooms',             # Bathroom count
    'category_id',           # 1=Res Sale, 2=Res Rent, 3=Comm Sale, 4=Comm Rent
    'property_type_id',      # Apartment, Villa, Office, etc.
    'days_active',           # Days since listing creation
    'distance_km',           # Haversine distance from seed/anchor listing
]
```

### Training Configuration

- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, English + Arabic)
- **XGBRanker:** `rank:ndcg`, 300 trees, depth 6, lr 0.02
- **User DNA:** KMeans(n_clusters=2) on weighted interaction vectors → dual-intent personas
- **Personalization blend:** 65% anchor vector + 35% closest user persona → search vector

### distance_km Feature (Geo-Aware Ranking)

The model uses geographic proximity as a ranking signal rather than a hard radius filter. This allows:

- No arbitrary distance cutoffs — model balances distance vs other relevance signals
- Cross-market adaptability — works across UAE, Egypt, Bahrain, KSA
- Automatic weight learning from user engagement data

**Training:** For each user, the listing with the highest interaction_score is the "anchor". Haversine distance is computed from anchor to every other listing the user interacted with.

**Serving:** In recsys_api, haversine distance is computed from the seed listing (the property being viewed) to each KNN candidate before sending to ranking_api.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Trained artifacts in `artifacts/` directory

### Run Services

```bash
# Start all services
docker-compose up -d --build

# Verify health
curl http://localhost:8002/health   # ranking_api
curl http://localhost:8001/health   # recsys_api
curl http://localhost:8000/health   # search_api
```

### Train the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Run full training pipeline
python sagemaker_pipeline.py

# Hot reload the new model (no restart needed)
curl -X POST http://localhost:8002/admin/reload-brain
```

### Seed Data

```bash
# Full seed (blue-green index swap)
python seed_data.py

# Delta seed (hourly upserts)
python seed_data.py --delta
```

## API Reference

### RecSys API — Recommendations (Port 8001)

```bash
# Get recommendations for a listing
GET /internal/v1/recommendations/{listing_id}?locale=en&limit=8

# With personalization (pass user ID in header)
GET /internal/v1/recommendations/{listing_id}?locale=en
  -H "X-PF-User-Id: user-abc-123"

# Filter by price period
GET /internal/v1/recommendations/{listing_id}?locale=en&filters.price_type=yearly
```

**Response:**
```json
{
  "properties": [
    { "id": "15592356", "rank_score": 0.683 },
    { "id": "1RXCVMYRNV", "rank_score": 0.448 }
  ]
}
```

### Search API — SERP (Port 8000)

```bash
# Basic search with ML ranking
GET /internal/v1/search/en?filters.category_id=2&filters.bedrooms=2&sorting.sort=mlWeights

# With price range
GET /internal/v1/search/en?filters.category_id=1&filters.min_price=500000&filters.max_price=2000000

# With geo filter
GET /internal/v1/search/en?filters.category_id=2&filters.latitude=25.08&filters.longitude=55.14&filters.radius_km=5

# With amenities
GET /internal/v1/search/en?filters.category_id=2&filters.amenities=Balcony,Private%20Pool

# Pagination
GET /internal/v1/search/en?filters.category_id=2&page.limit=20&page.number=2

# Sort modes: mlWeights, featured, newest, priceAsc, priceDesc, bedroomAsc, bedroomDesc, freshnessDate
```

**Response:**
```json
{
  "meta": { "page": 1, "total_count": 1234, "per_page": 50, "page_count": 25 },
  "properties": [{ "id": "...", "rank_score": 1.23 }],
  "fallback": false,
  "fallback_tier": null
}
```

**4-Tier Fallback:** If exact filters return 0 results, the API progressively relaxes: (1) drop amenities → (2) expand price ±20% → (3) expand price + drop geo filter.

### Ranking API — Model-as-a-Service (Port 8002)

```bash
# Rank a batch of properties
POST /rank
{
  "properties": [
    {
      "property_listing_id": "123",
      "price": 120000, "price_period": "yearly",
      "category_id": 2, "property_type_id": 1,
      "bedrooms": 2, "bathrooms": 2, "size_sqft": 1200,
      "listing_level": "featured", "quality_score": 0.7,
      "super_agent_score": 0.8, "popularity_score": 5.0,
      "listing_date": "2025-01-15",
      "distance_km": 12.5
    }
  ]
}

# Hot reload model
POST /admin/reload-brain

# Health check
GET /health
```

## Testing

```bash
# Unit tests (no services needed) — haversine math, feature engineering, training pipeline
python test_distance_km.py

# Integration tests (services must be running) — includes live API calls
python test_distance_km.py --integration

# Comprehensive quality suite (65+ checks) — geo, diversity, filters, sorts, edge cases, performance
python test_recsys_quality.py

# Property diversity test (29 archetypes) — checks recommendations across all property types
python test_diverse_properties.py

# Live API scenario tests — 11 real-world use cases
python test_api_scenarios.py
```

### Test Coverage Summary

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_distance_km.py` | 36 | Haversine math, vectorized ops, training pipeline, serving pipeline, live integration |
| `test_recsys_quality.py` | 65+ | Geo-awareness, personalization, diversity caps, ranking quality, all sort modes, all filters, fallback, edge cases, cross-market, cache, pagination, response structure, performance |
| `test_diverse_properties.py` | 29 | One test per property archetype (category × type × price tier × bedroom count) |
| `test_api_scenarios.py` | 11 | Affordable rental, luxury villa, commercial, cold start, studio, off-plan, land, short-term, amenity-rich |

## Production Results

| Metric | Value |
|--------|-------|
| Overall Relevance | 80% average across 29 archetypes |
| Category Accuracy | 100% |
| Type Match | 74% |
| Price Match | 58% |
| Bedroom Match | 73% |
| A-Grade Archetypes | 17/29 (59%) |
| F-Grade Archetypes | 0/29 (0%) |
| API Latency (RecSys) | < 2s |
| API Latency (Ranking 50 items) | < 500ms |
| Edge Case Pass Rate | 42/43 |

## Bug Fixes Applied

### Root Cause: XGBoost "Dead Matrix" (model returned 0.0 for ALL predictions)

Four cascading root causes were identified and fixed:

1. **RC-1: Label Variance Destruction** — `np.ceil()` crushed all interaction scores to 1. Fixed with `np.round(y * 10)` to preserve variance.

2. **RC-2: ID Format Mismatch** — Inventory used alphanumeric hashes, interactions used numeric web_ids. Only 6/6,133 listings overlapped. Fixed by standardizing on web_id for training joins.

3. **RC-3: Single-Item Query Groups** — 49 rows across 49 users meant 1 item per group. XGBRanker needs ≥2 items per group. Fixed by filtering out single-item groups.

4. **RC-4: Pricing Table Duplication** — Up to 92 rows per listing (83× bloat). Fixed with `ROW_NUMBER() OVER (PARTITION BY web_id)`.

### Additional Fixes

- Robust `listing_date` parsing (handles ISO strings and datetime objects)
- Post-training validation (predict on sample, flag if all 0.0)
- Comprehensive diagnostic logging at every pipeline stage
- Redis distributed lock with UUID-based atomic release
- Blue-green index swap with 90% success threshold
- Circuit breaker (Redis-shared, 5 failures → OPEN, 30s cooldown)

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Distance handling | Feature (not hard filter) | Model learns weight; no arbitrary cutoffs; works cross-market |
| Ranking deployment | Model-as-a-Service (1 worker) | brain.pkl in memory; asyncio.to_thread for concurrency; avoids OOM |
| Vector search | HNSW via OpenSearch | Native KNN; cosine similarity; same infra as filters |
| User DNA | Dual-intent (KMeans k=2) | Captures users who browse both rent + buy, or two neighborhoods |
| Personalization | 65/35 anchor/user blend | Anchor-dominant preserves relevance; user signal adds serendipity |
| ID system | Dual (web_id + property_listing_id) | web_id for training merge; property_listing_id for OpenSearch doc _id |
| Caching | Redis with TTL (5min anon, 2min personalized) | Balances freshness with load reduction |

## Environment Variables

| Variable | Default | Used By |
|----------|---------|---------|
| `OPENSEARCH_HOST` | `localhost` | search_api, recsys_api, seed_data |
| `REDIS_HOST` | `localhost` | search_api, recsys_api, seed_data |
| `RANKING_API_URL` | `http://ranking-api:8002` | search_api, recsys_api |
| `OPENSEARCH_INDEX` | `pf-inventory-v1` | all services |
| `BRAIN_PATH` | `artifacts/brain.pkl` | ranking_api |
| `ARTIFACTS_DIR` | `artifacts` | sagemaker_pipeline, seed_data |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | sagemaker_pipeline |
| `S3_BUCKET` | `pf-recsys-prod` | redshift_data |
| `REDSHIFT_CLUSTER_ID` | `pf-prod-cluster` | redshift_data |


