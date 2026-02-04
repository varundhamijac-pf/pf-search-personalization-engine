import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

# ==========================================
# 1. PRODUCTION CONFIGURATION
# ==========================================
app = FastAPI(title="Property Finder Recommendation Engine", version="6.6.0-SORTING-ENABLED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = {
    "database": pd.DataFrame(),
    "ltr_model": None,
    "features": [],
    "centroids": {},
    "status": "STARTING",
    "super_agent_threshold": 0.1 
}

# --- PROTO MAPPINGS ---
CATEGORY_MAP = { 1: "1", 2: "2", 3: "3", 4: "3", 5: "1" }
PROPERTY_TYPE_MAP = {
    1: "apartment", 35: "villa", 22: "townhouse",
    20: "penthouse", 24: "duplex", 5: "land", 4: "office"
}

FURNISHED_FILTER_MAP = { 1: ["1", "yes", "true"], 2: ["0", "no", "false"], 3: ["partly"] }
COMPLETION_FILTER_MAP = { 0: ["completed", "ready"], 1: ["off_plan", "offplan"], 2: ["off_plan", "offplan"] }

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def haversine_vectorized(lat1, lon1, lat2_series, lon2_series):
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_series)
    dphi = np.radians(lat2_series - lat1)
    dlambda = np.radians(lon2_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def safe_float(val, precision=2):
    try:
        val = float(val)
        if np.isnan(val) or np.isinf(val): return 0.0
        return round(val, precision)
    except:
        return 0.0

# ==========================================
# 3. LIFECYCLE (DATA LOADING)
# ==========================================
@app.on_event("startup")
async def startup_event():
    print("🚀 [INIT] Booting Recommendation Engine...")
    
    if os.path.exists('listings.parquet'):
        df = pd.read_parquet('listings.parquet')
        
        # Normalization
        df['offering_type'] = df['offering_type'].astype(str)
        df['property_type'] = df['property_type'].astype(str).str.lower()
        df['location_name'] = df['location_name'].astype(str)
        
        if 'furnished_flag' in df.columns:
            df['furnished_flag'] = df['furnished_flag'].astype(str).str.lower().replace({
                'yes': '1', 'furnished': '1', 'true': '1',
                'no': '0', 'unfurnished': '0', 'false': '0',
                'partly': 'partly', 'none': '0', 'nan': '0'
            })
            
        if 'completion_status' in df.columns:
            df['completion_status'] = df['completion_status'].astype(str).str.lower().replace({
                'completed_primary': 'completed',
                'off_plan_primary': 'off_plan',
                'ready': 'completed'
            })

        if 'is_verified' in df.columns:
            df['is_verified'] = df['is_verified'].astype(bool)
        else:
            df['is_verified'] = False

        if 'listing_date' in df.columns:
            df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
        
        numeric_defaults = ['quality_score', 'trust_score', 'smart_popularity_score', 'price', 'beds_int', 'bath_int', 'size_sqft']
        for col in numeric_defaults:
            if col not in df.columns: df[col] = 0.0
            else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Dynamic Threshold
        if 'trust_score' in df.columns and not df.empty:
            percentile_95 = df['trust_score'].quantile(0.95)
            if percentile_95 == 0:
                ENGINE['super_agent_threshold'] = 0.1
            else:
                ENGINE['super_agent_threshold'] = float(percentile_95)
        else:
            ENGINE['super_agent_threshold'] = 0.1

        valid_geo = df[(df['latitude'].notna()) & (df['latitude'] != 0)].copy()
        ENGINE['centroids'] = valid_geo.groupby('location_name')[['latitude', 'longitude']].mean().to_dict('index')
        ENGINE['database'] = df
        print(f"✅ [DATA] Loaded {len(df)} listings.")

    if os.path.exists('light_brain.pkl'):
        try:
            brain = joblib.load('light_brain.pkl')
            ENGINE['ltr_model'] = brain.get('ltr_model')
            ENGINE['features'] = brain.get('features', [])
            print(f"✅ [AI] Brain Loaded. Features: {len(ENGINE['features'])}")
        except Exception as e:
            print(f"❌ [CRITICAL] Brain Load Failed: {e}")

    ENGINE['status'] = "READY"

# ==========================================
# 4. REQUEST MODELS (UPDATED WITH SORTING)
# ==========================================
class Pagination(BaseModel):
    page: Optional[int] = 1
    limit: Optional[int] = 20

class NearBy(BaseModel):
    lat: float
    lon: float
    radius: float

# NEW: Sort Options matching .proto Enum
class SortBy(str, Enum):
    featured = "featured"
    newest = "newest"
    priceAsc = "priceAsc"
    priceDesc = "priceDesc"
    bedroomAsc = "bedroomAsc"
    bedroomDesc = "bedroomDesc"

# NEW: Sorting Object matching .proto Message
class Sorting(BaseModel):
    sort: Optional[SortBy] = SortBy.featured

class SearchFilters(BaseModel):
    keywords: Optional[List[str]] = [] 
    category_id: Optional[int] = None 
    property_type_ids: Optional[List[int]] = []
    min_price: Optional[int] = Field(None, alias="min_price")
    max_price: Optional[int] = Field(None, alias="max_price")
    number_of_bedrooms: Optional[List[int]] = []
    number_of_bathrooms: Optional[List[int]] = []
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    is_super_agent: Optional[bool] = False 
    is_verified: Optional[bool] = False
    furnished: Optional[int] = None        
    completion_status: Optional[int] = None 
    listed_within: Optional[int] = None    
    near_by: Optional[NearBy] = None

class ProtoSearchRequest(BaseModel):
    filters: Optional[SearchFilters] = None
    pagination: Optional[Pagination] = None
    sorting: Optional[Sorting] = None  # <--- Added Sorting Field

# ==========================================
# 5. SEARCH ENGINE LOGIC
# ==========================================
@app.post("/api/v1/search")
async def search(req: ProtoSearchRequest):
    if ENGINE['status'] != "READY":
        raise HTTPException(status_code=503, detail="Engine warming up")
        
    df = ENGINE['database']
    filters = req.filters or SearchFilters()
    mask = pd.Series([True] * len(df))
    search_method = "filters_only"
    
    threshold = ENGINE.get('super_agent_threshold', 0.1)

    # --- A. APPLY FILTERS ---
    if filters.category_id:
        val = CATEGORY_MAP.get(filters.category_id)
        if val: mask &= (df['offering_type'] == val)
    if filters.property_type_ids:
        types = [PROPERTY_TYPE_MAP.get(pid) for pid in filters.property_type_ids if PROPERTY_TYPE_MAP.get(pid)]
        if types: mask &= (df['property_type'].str.contains('|'.join(types), regex=True))
    if filters.min_price: mask &= (df['price'] >= filters.min_price)
    if filters.max_price: mask &= (df['price'] <= filters.max_price)
    if filters.min_area: mask &= (df['size_sqft'] >= filters.min_area)
    if filters.max_area: mask &= (df['size_sqft'] <= filters.max_area)
    if filters.number_of_bedrooms: mask &= (df['beds_int'].isin(filters.number_of_bedrooms))
    if filters.number_of_bathrooms: mask &= (df['bath_int'].isin(filters.number_of_bathrooms))
    if filters.is_super_agent:
        mask &= (df['trust_score'] >= threshold)
        search_method += "_trusted"
    if filters.is_verified:
        mask &= (df['is_verified'] == True)
    if filters.furnished and filters.furnished in FURNISHED_FILTER_MAP:
        allowed = FURNISHED_FILTER_MAP[filters.furnished]
        mask &= (df['furnished_flag'].isin(allowed))
    if filters.completion_status is not None and filters.completion_status in COMPLETION_FILTER_MAP:
        allowed = COMPLETION_FILTER_MAP[filters.completion_status]
        mask &= (df['completion_status'].isin(allowed))
    if filters.listed_within:
        cutoff = datetime.now() - timedelta(seconds=filters.listed_within)
        if 'listing_date' in df.columns:
            mask &= (df['listing_date'] >= cutoff)

    pool = df[mask].copy()

    # --- B. GEOSPATIAL SEARCH ---
    if filters.near_by:
        lat, lon, rad = filters.near_by.lat, filters.near_by.lon, filters.near_by.radius
        pool['dist_km'] = haversine_vectorized(lat, lon, pool['latitude'], pool['longitude'])
        pool = pool[pool['dist_km'] <= rad]
        search_method = f"geo_radius_{rad}km"
    elif filters.keywords and not pool.empty:
        raw_keyword = filters.keywords[0]
        search_term = raw_keyword.lower()
        matched_loc = next((k for k in ENGINE['centroids'] if k.lower() == search_term), None)
        if matched_loc:
            center = ENGINE['centroids'][matched_loc]
            pool['dist_km'] = haversine_vectorized(center['latitude'], center['longitude'], pool['latitude'], pool['longitude'])
            exact = pool[pool['dist_km'] <= 1.5]
            nearby = pool[pool['dist_km'] <= 3.5]
            if len(exact) > 0: pool = exact; search_method = "exact_radius_1.5km"
            elif len(nearby) > 0: pool = nearby; search_method = "nearby_radius_3.5km"
            else: pool = pool[pool['dist_km'] <= 10.0]; search_method = "wider_radius_10km"
        else:
            pool = pool[pool['location_name'].str.lower().str.contains(search_term, na=False)]
            search_method = "text_match"

    # --- C. AI SCORING ---
    if not pool.empty:
        now = datetime.now()
        if 'listing_date' in pool.columns:
            delta = (now - pool['listing_date']).dt.days.fillna(365).clip(lower=0)
            pool['freshness_score'] = 1 / (1 + (delta / 30.0))
        else:
            pool['freshness_score'] = 0.5 

        pool['ai_score'] = 0.0
        if ENGINE['ltr_model'] and ENGINE['features']:
            try:
                feats = ENGINE['features']
                model_input = pool.reindex(columns=feats, fill_value=0)
                if 'quality_score' in model_input.columns:
                    model_input['quality_score'] = pd.to_numeric(model_input['quality_score'], errors='coerce').fillna(0) / 250.0
                if 'furnished_flag' in model_input.columns:
                    model_input['furnished_flag'] = model_input['furnished_flag'].astype(str).str.lower().replace({
                        'yes': '1', 'true': '1', 'furnished': '1', 'partly': '1', 'no': '0', 'false': '0', 'unfurnished': '0', 'none': '0', 'nan': '0'
                    })
                if 'completion_status' in model_input.columns:
                    model_input['completion_status'] = model_input['completion_status'].astype(str).str.lower().replace({
                        'off_plan_primary': 'off_plan', 'ready': 'completed'
                    })
                cat_cols = ['offering_type', 'furnished_flag', 'property_type', 'location_id', 'completion_status']
                for col in cat_cols:
                    if col in model_input.columns:
                        model_input[col] = model_input[col].astype(str).replace(['nan', 'None'], 'unknown')
                        model_input[col] = model_input[col].astype('category')
                pool['ai_score'] = ENGINE['ltr_model'].predict_proba(model_input)[:, 1] * 100
            except Exception as e:
                print(f"⚠️ Inference Error: {e}")
                pool['ai_score'] = pool['freshness_score'] * 50

        if filters.min_price and filters.min_price >= 5000000:
            pool['ai_score'] += 10.0
            search_method += "_luxury"

        # Calculate default smart rank (used if no manual sort is selected)
        pool['final_rank'] = pool['ai_score'] - (pool.get('dist_km', 0) * 0.1)

    # --- D. SORTING LOGIC (NEW) ---
    # FIX: Added check 'if not pool.empty' to prevent KeyError on empty results
    if not pool.empty:
        sort_option = req.sorting.sort if (req.sorting and req.sorting.sort) else SortBy.featured
        
        if sort_option == SortBy.priceAsc:
            pool = pool.sort_values('price', ascending=True)
            search_method += "_sort_price_low"
        elif sort_option == SortBy.priceDesc:
            pool = pool.sort_values('price', ascending=False)
            search_method += "_sort_price_high"
        elif sort_option == SortBy.bedroomAsc:
            pool = pool.sort_values('beds_int', ascending=True)
            search_method += "_sort_beds_least"
        elif sort_option == SortBy.bedroomDesc:
            pool = pool.sort_values('beds_int', ascending=False)
            search_method += "_sort_beds_most"
        elif sort_option == SortBy.newest:
            if 'listing_date' in pool.columns:
                pool = pool.sort_values('listing_date', ascending=False)
                search_method += "_sort_newest"
        else:
            # Default: Featured (Smart AI Rank)
            if 'final_rank' in pool.columns:
                pool = pool.sort_values('final_rank', ascending=False)

    # --- E. RESPONSE ---
    limit = req.pagination.limit if req.pagination else 20
    page = req.pagination.page if req.pagination else 1
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    results = []
    # FIX: Safe iteration (will just do nothing if pool is empty)
    for _, row in pool.iloc[start_idx:end_idx].iterrows():
        img_url = row.get('image_url')
        if not img_url or pd.isna(img_url): img_url = "https://static.shared.propertyfinder.ae/media/images/listing/default.jpg"
        f_score = row.get('freshness_score', 0)
        is_super = bool(safe_float(row.get('trust_score')) >= threshold)
        
        results.append({
            "id": str(row['property_listing_id']),
            "title": row.get('property_title', 'Listing'),
            "location": {
                "name": str(row.get('location_name', 'Unknown')),
                "lat": safe_float(row.get('latitude')),
                "lon": safe_float(row.get('longitude'))
            },
            "price": {
                "value": safe_float(row.get('price')),
                "currency": "AED",
                "period": "rent" if str(row.get('offering_type')) == "2" else "sell"
            },
            "bedrooms": { "value": int(safe_float(row.get('beds_int'))) },
            "images": [{ "medium": img_url }],
            "tags": [search_method, f"dist: {safe_float(row.get('dist_km', 0), 2)}km"],
            "scores": {
                "ai_match_percentage": safe_float(row.get('ai_score', 0), 1),
                "freshness_boost": "Active" if f_score >= 0.7 else "Standard"
            },
            "agent": {
                "name": row.get('agent_name', "Property Consultant"),
                "is_super_agent": is_super
            }
        })

    return {
        "meta": { "total_count": len(pool), "page": page, "search_method": search_method },
        "properties": results
    }