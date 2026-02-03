import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
app = FastAPI(title="Property Finder Hybrid AI Engine", version="2.0.0")

# CORS: Allow all for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
ENGINE = {
    "database": pd.DataFrame(),
    "ltr_model": None,
    "features": []
}

# Standard mappings
OFFERING_MAP = {"sale": "1", "buy": "1", "rent": "2", "commercial": "3"}

# ==========================================
# 2. LIFECYCLE EVENTS
# ==========================================
@app.on_event("startup")
async def startup_event():
    print("⏳ [System] Initializing Engine...")
    
    # A. Load the Brain
    try:
        if os.path.exists('light_brain.pkl'):
            brain = joblib.load('light_brain.pkl')
            ENGINE['ltr_model'] = brain['ltr_model']
            ENGINE['features'] = brain.get('features', [])
            print(f"✅ [System] AI Brain Loaded. Features: {len(ENGINE['features'])}")
        else:
            print("⚠️ [Warning] AI Brain not found. Running in heuristic mode.")
    except Exception as e:
        print(f"❌ [Critical] Brain Load Failed: {e}")

    # B. Load the Body
    try:
        if os.path.exists('listings.parquet'):
            df = pd.read_parquet('listings.parquet')
            if 'listing_date' in df.columns:
                df['listing_date'] = pd.to_datetime(df['listing_date'])
            ENGINE['database'] = df
            print(f"✅ [System] Database Loaded: {len(df)} listings.")
        else:
            print("❌ [Critical] listings.parquet not found!")
    except Exception as e:
        print(f"❌ [Critical] Data Load Failed: {e}")

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def calculate_freshness(dates):
    now = datetime.now()
    delta = (now - dates).dt.days
    delta = np.maximum(delta, 0)
    return 1 / (1 + (delta / 30.0))

def haversine_vectorized(lat1, lon1, lat_series, lon_series):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat_series)
    dphi = np.radians(lat_series - lat1)
    dlambda = np.radians(lon_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# --- NEW: SAFETY HELPER ---
def safe_float(val, precision=1):
    """Converts NaN/Inf to 0.0 to prevent JSON crashes"""
    try:
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return round(val, precision)
    except:
        return 0.0

# ==========================================
# 4. API REQUEST MODELS
# ==========================================
class UserProfile(BaseModel):
    target_price: Optional[float] = None
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    persona: Optional[str] = "guest"

class SearchRequest(BaseModel):
    query: Optional[str] = ""
    location_name: str = ""
    min_price: float = 0
    max_price: float = 0
    bedrooms: str = "-1"
    offering_type: str = "sale"
    property_type: str = "unknown"
    limit: int = 20
    sort_by: str = "relevance"
    user_profile: Optional[UserProfile] = None

# ==========================================
# 5. RECOMMENDATION ENDPOINT
# ==========================================
@app.post("/api/v1/recommend")
async def recommend(req: SearchRequest):
    df = ENGINE['database']
    if df.empty: return {"status": "error", "message": "Database not ready"}

    # --- STEP 1: STRICT FILTERING ---
    mask = pd.Series([True] * len(df))

    req_off = OFFERING_MAP.get(req.offering_type.lower(), None)
    if req_off:
        mask &= (df['offering_type'].astype(str) == req_off)

    if req.property_type not in ['unknown', 'all']:
        mask &= df['property_type'].str.lower().str.contains(req.property_type.lower())

    try:
        target_beds = int(str(req.bedrooms).split()[0])
        if target_beds >= 0:
            mask &= (df['beds_int'] == target_beds)
    except: pass

    if req.min_price > 0: mask &= (df['price'] >= req.min_price)
    if req.max_price > 0: mask &= (df['price'] <= req.max_price)

    pool = df[mask].copy()
    if pool.empty: return {"status": "success", "count": 0, "results": [], "message": "No strict matches"}

    # --- STEP 2: GEOSPATIAL WATERFALL ---
    search_radius_tag = "Text Match"
    
    if req.location_name:
        loc_match = df[df['location_name'].str.contains(req.location_name, case=False, na=False)]
        
        if not loc_match.empty:
            center_lat = loc_match.iloc[0]['latitude']
            center_lon = loc_match.iloc[0]['longitude']
            
            pool['dist_km'] = haversine_vectorized(center_lat, center_lon, pool['latitude'], pool['longitude'])
            
            b1 = pool[pool['dist_km'] <= 1.5]
            b2 = pool[pool['dist_km'] <= 3.5]
            b3 = pool[pool['dist_km'] <= 10.0]
            
            if len(b1) >= 3:
                candidates = b1.copy(); search_radius_tag = "Exact Location (<1.5km)"
            elif len(b2) >= 3:
                candidates = b2.copy(); search_radius_tag = "Nearby Areas (<3.5km)"
            elif len(b3) >= 3:
                candidates = b3.copy(); search_radius_tag = "Wider Region (<10km)"
            else:
                candidates = pool.sort_values('dist_km').head(50).copy(); search_radius_tag = "Closest Matches"
        else:
            candidates = pool[pool['location_name'].str.contains(req.location_name, case=False, na=False)].copy()
            candidates['dist_km'] = 0
    else:
        candidates = pool.copy()
        candidates['dist_km'] = 0

    if candidates.empty: return {"status": "success", "count": 0, "results": []}

    # --- STEP 3: AI SCORING ---
    candidates['freshness_score'] = calculate_freshness(candidates['listing_date'])
    
    if ENGINE['ltr_model']:
        try:
            feats = ENGINE['features']
            for f in feats:
                if f not in candidates.columns: candidates[f] = 0
            
            X_input = candidates[feats].fillna(0)
            candidates['ai_score'] = ENGINE['ltr_model'].predict_proba(X_input)[:, 1] * 100
        except:
            candidates['ai_score'] = 50.0
    else:
        candidates['ai_score'] = 0.0

    candidates['geo_score'] = (1 / (1 + candidates['dist_km'])) * 100
    candidates['final_score'] = (candidates['ai_score'] * 0.6) + (candidates['geo_score'] * 0.4)

    # --- STEP 4: PERSONALIZATION ---
    profile = req.user_profile
    if profile and profile.target_price:
        price_sim = np.exp(- (np.abs(candidates['price'] - profile.target_price)**2) / (2 * 500000**2)) * 100
        candidates['p_score'] = (candidates['final_score'] * 0.6) + (price_sim * 0.4)
    elif profile and profile.persona == 'luxury':
        candidates['p_score'] = candidates['final_score'] + (candidates['price'] / 1000000)
    elif profile and profile.persona == 'budget':
        candidates['p_score'] = candidates['final_score'] + (1000000 / (candidates['price'] + 1))
    else:
        candidates['p_score'] = candidates['final_score']

    # --- STEP 5: SORT & FORMAT (With NaN Protection) ---
    sort_mode = req.sort_by.lower()
    if sort_mode == 'price_asc':
        final_df = candidates.sort_values('price', ascending=True)
    elif sort_mode == 'newest':
        final_df = candidates.sort_values('freshness_score', ascending=False)
    else:
        final_df = candidates.sort_values('p_score', ascending=False)

    results = []
    for _, row in final_df.head(req.limit).iterrows():
        # Safe Image Handling
        img = row.get('image_url')
        if not img or str(img).lower() == 'nan' or str(img) == '': 
            img = "https://static.shared.propertyfinder.ae/media/images/listing/default.jpg"

        # Safe Agent Data
        agent_trust = safe_float(row.get('trust_score', 0), 2)
        agent_info = {
            "name": row.get('agent_name', "Property Consultant"),
            "is_super": bool(agent_trust > 0.7),
            "score": round(agent_trust * 100, 1)
        }

        # Safe Coordinates
        lat = safe_float(row.get('latitude', 0), 6)
        lon = safe_float(row.get('longitude', 0), 6)

        results.append({
            "id": str(row['property_listing_id']),
            "title": row.get('property_title', 'Listing'),
            "location": {
                "name": str(row.get('location_name', 'Unknown')),
                "lat": lat,
                "lon": lon
            },
            # Use safe_float for all numerical outputs to prevent crashes
            "price": safe_float(row['price'], 0),
            "specs": {
                "beds": int(safe_float(row.get('beds_int', 0), 0)),
                "baths": int(safe_float(row.get('bath_int', 0), 0)),
                "area": int(safe_float(row.get('size_sqft', 0), 0))
            },
            "scores": {
                "ai_match": safe_float(row.get('ai_score', 0), 1),
                "quality": int(safe_float(row.get('quality_score', 0), 0)), 
                "freshness": safe_float(row.get('freshness_score', 0) * 100, 1)
            },
            "media": {
                "image": img,
                "video": row.get('video_url', None)
            },
            "agent": agent_info,
            "tags": [search_radius_tag],
            "verified": bool(row.get('is_verified', False))
        })

    return {
        "status": "success",
        "meta": {
            "count": len(results),
            "radius_logic": search_radius_tag
        },
        "results": results
    }