import pandas as pd
import joblib
import numpy as np
import os
import boto3
import logging
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum, IntEnum
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. CONFIGURATION & LOGGING ---
app = FastAPI(
    title="Property Finder RecSys (World Class)", 
    version="12.4-Production",
    description="High-performance recommendation engine with XGBoost and NLP Personalization"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

# AWS Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "your-s3-bucket-name")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. ENUMS & CONSTANTS ---
class PropertyType(IntEnum):
    APARTMENT = 1
    VILLA = 35
    TOWNHOUSE = 22
    PENTHOUSE = 20
    DUPLEX = 24
    HOTEL_APARTMENT = 45 
    COMPOUND = 42
    WHOLE_BUILDING = 10
    FULL_FLOOR = 18
    RESIDENTIAL_PLOT = 19
    OFFICE_SPACE = 4
    WAREHOUSE = 13
    SHOW_ROOM = 12
    SHOP = 21
    RETAIL = 27
    LABOR_CAMP = 11
    STAFF_ACCOMMODATION = 43
    FACTORY = 45 
    BULK_SALE_UNIT = 30
    BULK_RENT_UNIT = 34
    LAND = 19

class Category(IntEnum):
    UNSPECIFIED = 0
    RESIDENTIAL_SALE = 1
    RESIDENTIAL_RENT = 2
    COMMERCIAL_SALE = 3
    COMMERCIAL_RENT = 4
    NEW_PROJECTS = 5

class CompletionStatus(IntEnum):
    COMPLETED = 0
    OFF_PLAN = 1

class Furnished(IntEnum):
    ALL = 0
    YES = 1
    NO = 2
    PARTLY = 3

class PaymentMethod(IntEnum):
    ALL_METHODS = 0
    INSTALLMENTS = 1
    CASH = 2

class RentalPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

class VirtualViewings(IntEnum):
    ANY = 0
    D360 = 1
    VIDEO = 2
    LIVE = 3

class SortBy(str, Enum):
    ML_WEIGHTS = "mlWeights" 
    FEATURED = "featured"
    NEWEST = "newest"
    PRICE_ASC = "priceAsc"
    PRICE_DESC = "priceDesc"
    BEDROOM_ASC = "bedroomAsc"
    BEDROOM_DESC = "bedroomDesc"
    DISTANCE_ASC = "distanceAsc"

class Currency(str, Enum):
    AED = "AED"
    USD = "USD"
    EGP = "EGP"
    SAR = "SAR"
    BHD = "BHD"

# Constant for identifying commercial units to handle "0-bedroom" display logic
COMMERCIAL_IDS = [4, 5, 10, 11, 12, 13, 19, 21, 27, 29, 34, 42, 43, 44, 45]

# --- 3. GEOSPATIAL UTILS ---
def haversine_vectorized(lat1, lon1, lat2_series, lon2_series):
    R = 6371.0 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_series)
    dphi = np.radians(lat2_series - lat1)
    dlambda = np.radians(lon2_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- 4. REQUEST MODELS ---
class SearchFilters(BaseModel):
    category_id: Optional[Category] = Field(None, alias="offering_type")
    property_type_ids: Optional[List[int]] = []
    number_of_bedrooms: Optional[List[int]] = Field([], alias="bedrooms")
    number_of_bathrooms: Optional[List[int]] = Field([], alias="bathrooms")
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_area: Optional[float] = None
    max_area: Optional[float] = None
    completion_status: Optional[CompletionStatus] = None 
    furnished: Optional[Furnished] = None
    rental_period: Optional[RentalPeriod] = None
    payment_method: Optional[PaymentMethod] = None
    is_verified: Optional[bool] = None
    is_super_agent: Optional[bool] = None
    new_projects: Optional[bool] = None
    keywords: Optional[List[str]] = []
    amenities: Optional[List[str]] = []
    locations_ids: Optional[List[str]] = Field([], alias="location_ids")
    virtual_viewings: Optional[VirtualViewings] = None 
    polygon: Optional[List[str]] = None
    bounding_box: Optional[List[str]] = None
    travel_time: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = 10.0

    model_config = ConfigDict(populate_by_name=True)

class Pagination(BaseModel):
    page: int = 1
    limit: int = 50

class Sorting(BaseModel):
    sort: Optional[SortBy] = SortBy.ML_WEIGHTS

class SearchRequest(BaseModel):
    user_id: Optional[str] = None
    filters: SearchFilters
    pagination: Optional[Pagination] = Pagination() 
    sorting: Optional[Sorting] = Sorting()

# --- 5. RESPONSE MODELS ---
class ProtoImage(BaseModel):
    small: Optional[str] = None
    medium: Optional[str] = None

class ProtoAgent(BaseModel):
    id: str
    name: str
    image: Optional[str] = None
    is_super_agent: bool = False

class ProtoBroker(BaseModel):
    id: str
    name: Optional[str] = None
    logo: Optional[str] = None

class ProtoPrice(BaseModel):
    value: float
    currency: Currency = Currency.AED
    period: Optional[str] = None

class Coordinates(BaseModel):
    lat: float
    lon: float

class ProtoLocation(BaseModel):
    id: str
    full_name: str
    coordinates: Coordinates

class ProtoProperty(BaseModel):
    id: str
    category_id: int
    property_type: str
    price: ProtoPrice
    location: ProtoLocation
    title: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    size_sqft: Optional[float] = None
    amenities: List[str] = []
    images: List[ProtoImage] = []
    agent: Optional[ProtoAgent] = None
    broker: Optional[ProtoBroker] = None
    is_verified: bool = False
    is_premium: bool = False
    is_featured: bool = False
    completion_status: Optional[str] = None
    furnished: Optional[str] = None
    rank_score: float
    distance_km: Optional[float] = None

class SearchResponse(BaseModel):
    meta: Dict[str, Any]
    properties: List[ProtoProperty]

class AggregationBucket(BaseModel):
    id: str
    count: int

class AggregationsResponse(BaseModel):
    bedrooms: List[AggregationBucket] = []
    property_types: List[AggregationBucket] = []

# --- 6. STATE & HELPERS ---
BRAIN = None
INVENTORY = None
USER_STORE = None
VECTORIZER = None

def amenity_tokenizer(text):
    return str(text).split(',')

# --- 7. STARTUP & DATA LOADING ---
def download_from_s3(client, key, local_path):
    try:
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET}/{key}...")
        client.download_file(S3_BUCKET, key, local_path)
        return True
    except Exception as e:
        logger.warning(f"⚠️ S3 Download Failed: {e}")
        return False

@app.on_event("startup")
def startup_event():
    global BRAIN, INVENTORY, USER_STORE, VECTORIZER
    
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        artifacts = ["inventory.parquet", "user_data.parquet", "brain.pkl"]
        for art in artifacts:
            if not os.path.exists(art):
                download_from_s3(s3, f"artifacts/{art}", art)
    except Exception: 
        logger.info(" AWS connection skipped, looking for local files.")

    try:
        if os.path.exists("brain.pkl"):
            BRAIN = joblib.load("brain.pkl")
            if BRAIN.get('vocab'):
                # Added explicit token_pattern=None for token consistency
                VECTORIZER = CountVectorizer(tokenizer=amenity_tokenizer, token_pattern=None, vocabulary=BRAIN['vocab'])
            logger.info(" Brain loaded successfully.")

        if os.path.exists("inventory.parquet"):
            INVENTORY = pd.read_parquet("inventory.parquet")
            
            if BRAIN and 'features' in BRAIN:
                for col in BRAIN['features']:
                    if col in INVENTORY.columns:
                        INVENTORY[col] = pd.to_numeric(INVENTORY[col], errors='coerce').fillna(0.0)
            
            if 'bedrooms_int' in INVENTORY.columns and 'bedrooms' not in INVENTORY.columns:
                INVENTORY.rename(columns={'bedrooms_int': 'bedrooms'}, inplace=True)

            INVENTORY['category_id'] = INVENTORY.get('category_id', 0).fillna(0).astype(int)
            INVENTORY['property_type_id'] = INVENTORY.get('property_type_id', 0).fillna(0).astype(int)
            
            text_cols = ['location_name', 'full_location_path', 'property_title', 'amenities']
            for col in text_cols:
                if col in INVENTORY.columns:
                    INVENTORY[col] = INVENTORY[col].astype(str).str.replace('nan', '', case=False, regex=False)

            logger.info(f" Inventory Online: {len(INVENTORY)} listings")
        
        if os.path.exists("user_data.parquet"):
            USER_STORE = pd.read_parquet("user_data.parquet")

    except Exception as e:
        logger.error(f" Startup Failure: {e}")

# --- 8. SEARCH ENDPOINT ---
@app.post("/property-api/v1/en/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    if INVENTORY is None: raise HTTPException(status_code=503, detail="System Booting")
    
    f = req.filters
    df = INVENTORY.copy()
    
    # --- A. Smart Defaults & Category Rules ---
    is_default_yearly = False
    if f.category_id in [Category.RESIDENTIAL_RENT, Category.COMMERCIAL_RENT]:
        if not f.rental_period:
            f.rental_period = RentalPeriod.YEARLY
            is_default_yearly = True
    if f.category_id in [Category.RESIDENTIAL_SALE, Category.COMMERCIAL_SALE]:
        f.rental_period = None

    # --- B. Hard Filtering ---
    if f.category_id: 
        df = df[df['category_id'] == f.category_id.value]
    
    if f.property_type_ids: 
        df = df[df['property_type_id'].isin(f.property_type_ids)]
        
    if f.min_price: df = df[df['price'] >= f.min_price]
    if f.max_price: df = df[df['price'] <= f.max_price]
    if f.min_area: df = df[df['size_sqft'] >= f.min_area]
    if f.max_area: df = df[df['size_sqft'] <= f.max_area]

    if f.rental_period:
        if f.rental_period == RentalPeriod.YEARLY and is_default_yearly:
            df = df[df['price_period'].astype(str).str.lower().isin(['yearly', 'nan', '', None]) | df['price_period'].isna()]
        else:
            df = df[df['price_period'].astype(str).str.lower() == f.rental_period.value]
        
    if f.locations_ids:
        df = df[df['location_id'].astype(str).apply(lambda x: x.split('.')[0]).isin(f.locations_ids)]

    if f.latitude and f.longitude:
        df['distance_km'] = haversine_vectorized(f.latitude, f.longitude, df['latitude'], df['longitude'])
        df = df[df['distance_km'] <= f.radius_km]
    else:
        df['distance_km'] = 0.0

    if f.number_of_bedrooms:
        if 7 in f.number_of_bedrooms:
            df = df[df['bedrooms'].isin(f.number_of_bedrooms) | (df['bedrooms'] >= 7)]
        else:
            df = df[df['bedrooms'].isin(f.number_of_bedrooms)]

    if f.number_of_bathrooms:
        if 7 in f.number_of_bathrooms:
            df = df[df['bathrooms'].isin(f.number_of_bathrooms) | (df['bathrooms'] >= 7)]
        else:
            df = df[df['bathrooms'].isin(f.number_of_bathrooms)]

    if f.keywords:
        pattern = '|'.join([re.escape(k.strip()) for k in f.keywords])
        mask = (
            df['property_title'].str.contains(pattern, case=False, na=False) | 
            df['location_name'].str.contains(pattern, case=False, na=False) |
            df['full_location_path'].str.contains(pattern, case=False, na=False)
        )
        df = df[mask]
    
    if f.amenities:
        for am in f.amenities:
            df = df[df['amenities'].str.contains(am, case=False, na=False)]

    if f.virtual_viewings and f.virtual_viewings != VirtualViewings.ANY:
        if f.virtual_viewings == VirtualViewings.VIDEO:
             if 'video_url' in df.columns: df = df[df['video_url'].notna() & (df['video_url'] != '')]
        elif f.virtual_viewings == VirtualViewings.D360:
             if 'view_360' in df.columns: df = df[df['view_360'].notna() & (df['view_360'] != '')]

    if df.empty: return {"meta": {"count": 0, "page": 1, "scoring": "ML_v12.4_Production"}, "properties": []}

    # --- C. AI SCORING & UPDATED PERSONALIZATION ---
    df['rank_score'] = 0.0
    if BRAIN:
        try:
            X_input = df.reindex(columns=BRAIN['features'], fill_value=0.0)
            df['rank_score'] = BRAIN['model'].predict(X_input)
            df['rank_score'] = df['rank_score'].clip(lower=0)
            
            # --- Personalization Layer ---
            if req.user_id and USER_STORE is not None and VECTORIZER:
                user_hist = USER_STORE[USER_STORE['user_id'] == req.user_id]
                if not user_hist.empty:
                    liked_ids = user_hist['property_listing_id'].unique()
                    liked_df = INVENTORY[INVENTORY['property_listing_id'].isin(liked_ids)]
                    if not liked_df.empty:
                        # Extract preferences and create preference vector
                        user_vector = np.asarray(VECTORIZER.transform(liked_df['amenities'].fillna('')).mean(axis=0))
                        # Transform currently filtered properties
                        candidate_vectors = VECTORIZER.transform(df['amenities'].fillna(''))
                        # Calculate dot product similarity
                        similarity = np.asarray(candidate_vectors.dot(user_vector.T)).flatten()
                        # Apply personalization boost (5.0 weight)
                        df['rank_score'] += (similarity * 5.0) 
        except Exception as e:
            logger.error(f"Scoring Component Failed: {e}")

    # --- D. SORTING ---
    sort_choice = req.sorting.sort
    if sort_choice == SortBy.PRICE_ASC: df = df.sort_values("price", ascending=True)
    elif sort_choice == SortBy.PRICE_DESC: df = df.sort_values("price", ascending=False)
    elif sort_choice == SortBy.DISTANCE_ASC: df = df.sort_values("distance_km", ascending=True)
    elif sort_choice == SortBy.NEWEST and 'listing_date' in df.columns: df = df.sort_values("listing_date", ascending=False)
    elif sort_choice == SortBy.BEDROOM_ASC: df = df.sort_values("bedrooms", ascending=True)
    elif sort_choice == SortBy.BEDROOM_DESC: df = df.sort_values("bedrooms", ascending=False)
    else: df = df.sort_values("rank_score", ascending=False)

    # --- E. PAGINATION & RESPONSE ---
    total_found = len(df)
    page_start = (req.pagination.page - 1) * req.pagination.limit
    page_end = req.pagination.page * req.pagination.limit
    page_df = df.iloc[page_start : page_end]

    results = []
    for _, row in page_df.iterrows():
        tid = int(row.get('property_type_id', 0))
        beds = int(row.get('bedrooms', 0))
        # Commercial logic: Don't show "studio" for warehouses/offices
        bed_display = "0" if tid in COMMERCIAL_IDS else ("studio" if beds == 0 else str(beds))

        img_raw = str(row.get('images', ''))
        images = [ProtoImage(medium=u.strip(), small=u.strip()) for u in img_raw.split(',') if u.strip()][:5]

        results.append(ProtoProperty(
            id=str(row['property_listing_id']),
            category_id=int(row['category_id']),
            property_type=str(row.get('property_type', 'Unknown')),
            price=ProtoPrice(value=float(row['price']), currency=Currency.AED, period=row.get('price_period')),
            location=ProtoLocation(
                id=str(row.get('location_id', '0')),
                full_name=str(row.get('location_name', '')),
                coordinates=Coordinates(lat=float(row['latitude']), lon=float(row['longitude']))
            ),
            title=str(row.get('property_title', '')),
            bedrooms=bed_display,
            bathrooms=str(int(row.get('bathrooms', 0))),
            size_sqft=float(row.get('size_sqft', 0)),
            amenities=[a.strip() for a in str(row.get('amenities', '')).split(',') if a.strip()],
            images=images,
            agent=ProtoAgent(
                id=str(row.get('agent_id', '0')), 
                name=str(row.get('agent_name', 'Unknown')),
                is_super_agent=bool(row.get('super_agent_score', 0) > 0)
            ),
            broker=ProtoBroker(
                id=str(row.get('broker_id', '0')), 
                name=str(row.get('broker_name', 'Unknown')),
                logo=str(row.get('broker_logo', ''))
            ),
            is_verified=bool(row.get('pending_verified_flag', 0) > 0),
            is_premium=bool(str(row.get('listing_level', '')).lower() == 'premium'),
            is_featured=bool(str(row.get('listing_level', '')).lower() == 'featured'),
            rank_score=float(row['rank_score']),
            distance_km=float(row['distance_km'])
        ))

    return {
        "meta": {"page": req.pagination.page, "total_count": total_found, "scoring": "ML_v12.4_Production"},
        "properties": results
    }

# --- 9. AGGREGATIONS ENDPOINT ---
@app.post("/property-api/v1/en/aggs", response_model=AggregationsResponse)
async def get_aggregations(req: SearchRequest):
    if INVENTORY is None: return {"bedrooms": [], "property_types": []}
    df = INVENTORY.copy()
    if req.filters.category_id:
        df = df[df['category_id'] == req.filters.category_id.value]
    if req.filters.locations_ids:
        df = df[df['location_id'].astype(str).apply(lambda x: x.split('.')[0]).isin(req.filters.locations_ids)]

    bed_buckets = []
    if not df.empty and 'bedrooms' in df.columns:
        counts = df['bedrooms'].value_counts().sort_index().to_dict()
        for b, c in counts.items():
            is_comm = req.filters.category_id in [Category.COMMERCIAL_SALE, Category.COMMERCIAL_RENT]
            label = "0" if is_comm and b == 0 else ("studio" if b == 0 else str(int(b)))
            bed_buckets.append(AggregationBucket(id=label, count=c))
            
    prop_buckets = []
    if not df.empty and 'property_type_id' in df.columns:
        t_counts = df['property_type_id'].value_counts().to_dict()
        for tid, count in t_counts.items():
            prop_buckets.append(AggregationBucket(id=str(int(tid)), count=count))

    return {"bedrooms": bed_buckets, "property_types": prop_buckets}

# --- 10. HEALTHCHECK ---
@app.get("/property-api/healthcheck")
def health():
    return {
        "status": "ok", 
        "inventory_size": len(INVENTORY) if INVENTORY is not None else 0,
        "brain_online": BRAIN is not None,
        "personalization_online": VECTORIZER is not None
    }

# --- 11. SERVER ENTRY POINT ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)