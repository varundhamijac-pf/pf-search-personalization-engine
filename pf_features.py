import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 1. CANONICAL FEATURE LIST — used by XGBRanker train AND serve
# ═══════════════════════════════════════════════════════════════════════
MODEL_FEATURES = [
    'feature_annual_rent', 'feature_sale_price', 'is_sale', 'price_per_sqft',
    'listing_level_score', 'super_agent_score', 'popularity_score',
    'bedrooms', 'bathrooms', 'category_id', 'property_type_id', 'days_active',
    'distance_km',  # NEW: Haversine distance from seed to candidate
]
# ═══════════════════════════════════════════════════════════════════════
# 2. AMENITY MAP — reverse lookup for filter-based SERP queries
# ═══════════════════════════════════════════════════════════════════════
AMENITY_MAP = {
    'Maids Room': 1, 'Study': 2, 'Central A/C': 3, 'Balcony': 4,
    'Private Garden': 5, 'Private Pool': 6, 'Private Gym': 7,
    'Private Jacuzzi': 8, 'Shared Pool': 9, 'Shared Spa': 10,
    'Security': 11, 'Concierge': 12, 'Maid Service': 13,
    'Covered Parking': 14, 'Built in Wardrobes': 15, 'Walk-in Closet': 16,
    'Kitchen Appliances': 17, 'View of Water': 18, 'View of Landmark': 19,
    'Pets Allowed': 20, 'Shared Gym': 24, 'Lobby in Building': 25,
    'Pantry': 26, 'Barbecue Area': 29, 'Mezzanine': 30, 'Lift': 32,
    "Driver's room": 33
}

AMENITY_ID_TO_NAME = {v: k for k, v in AMENITY_MAP.items()}

# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — single definition, used everywhere
# ═══════════════════════════════════════════════════════════════════════
RENT_MULTIPLIER_MAP = {'daily': 365, 'weekly': 52, 'monthly': 12, 'yearly': 1, 'sell': 1}
LISTING_LEVEL_MAP = {'premium': 2, 'featured': 1, 'standard': 0}

def preprocess_for_model(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Feature engineering function used IDENTICALLY during training AND serving.
    """
    df = df.copy()

    # ── is_sale: category_id 1 (Residential Sale) or 3 (Commercial Sale) ──
    df['category_id'] = pd.to_numeric(df.get('category_id', pd.Series(dtype=float)), errors='coerce').fillna(0)
    df['is_sale'] = df['category_id'].isin([1, 3]).astype(int)

    # ── Annual Rent / Sale Price ──
    df['rent_multiplier'] = (
        df['price_period'].astype(str).str.lower().str.strip()
        .map(RENT_MULTIPLIER_MAP).fillna(1)
    )
    price_num = pd.to_numeric(df.get('price', pd.Series(dtype=float)), errors='coerce').fillna(0)
    df['feature_annual_rent'] = np.where(df['is_sale'] == 0, price_num * df['rent_multiplier'], 0.0)
    df['feature_sale_price'] = np.where(df['is_sale'] == 1, price_num, 0.0)

    # ── Price per sqft ──
    safe_sqft = pd.to_numeric(df.get('size_sqft', pd.Series(dtype=float)), errors='coerce').replace(0, 1).fillna(1)
    df['price_per_sqft'] = (df['feature_annual_rent'] + df['feature_sale_price']) / safe_sqft

    # ── Listing level score ──
    df['listing_level_score'] = (
        df.get('listing_level', pd.Series(dtype=str)).astype(str).str.lower().str.strip()
        .map(LISTING_LEVEL_MAP).fillna(0)
    )

    # ── Days active ──
    raw_dates = df.get('listing_date', pd.Series(dtype='object'))
    parsed_dates = pd.to_datetime(raw_dates, errors='coerce', format='mixed', dayfirst=False)
    
    now = pd.Timestamp.now()
    days = (now - parsed_dates).dt.days
    df['days_active'] = days.clip(lower=0).fillna(30)

    # ── Ensure all MODEL_FEATURES exist and are numeric ──
    # Note: distance_km will automatically be filled with 0.0 if missing (e.g., during training)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if verbose:
        logger.info("=== PREPROCESS DIAGNOSTIC ===")
        for col in MODEL_FEATURES:
            vals = df[col]
            non_zero = (vals != 0).sum()
            logger.info(
                f"  {col:25s}: min={vals.min():.4f}, max={vals.max():.4f}, "
                f"mean={vals.mean():.4f}, non_zero={non_zero}/{len(vals)}"
            )

    return df

# ═══════════════════════════════════════════════════════════════════════
# 4. RICH DESCRIPTION — for sentence-transformer embedding input
# ═══════════════════════════════════════════════════════════════════════
PRICE_SEGMENT_LABELS = [
    (0, 'budget'), (500_000, 'mid-range'), (2_000_000, 'premium'),
    (5_000_000, 'luxury'), (15_000_000, 'ultra-luxury'),
]

def _price_segment(price: float) -> str:
    label = 'budget'
    for threshold, seg in PRICE_SEGMENT_LABELS:
        if price >= threshold: label = seg
    return label

def _resolve_amenity_names(raw_amenities) -> str:
    if raw_amenities is None: return ''
    if isinstance(raw_amenities, float) and pd.isna(raw_amenities): return ''
        
    names = []
    if isinstance(raw_amenities, list):
        for p in raw_amenities:
            try:
                name = AMENITY_ID_TO_NAME.get(int(p))
                if name: names.append(name)
            except (ValueError, TypeError): pass
    else:
        parts = str(raw_amenities).split(',')
        for p in parts:
            p = p.strip()
            if p.isdigit():
                name = AMENITY_ID_TO_NAME.get(int(p))
                if name: names.append(name)
            elif p: names.append(p)
                
    return ', '.join(names[:8])

def build_rich_description(row) -> str:
    beds = str(row.get('bedrooms_label', row.get('bedrooms', '0')))
    if beds in ('0', '0.0', '') or 'studio' in str(beds).lower(): beds = 'Studio'
    else: beds = f"{int(float(beds))} Bedroom" if beds.replace('.', '').isdigit() else beds

    prop_type = str(row.get('property_type', 'Property'))
    location = str(row.get('location_name', ''))
    full_path = str(row.get('full_location_path', location))

    price = float(row.get('price', 0))
    segment = _price_segment(price)
    period = str(row.get('price_period', 'sell')).lower()
    price_desc = f"{segment} {period}" if period != 'sell' else f"{segment} for sale"

    amenities = _resolve_amenity_names(row.get('amenities', ''))

    parts = [f"{beds} {prop_type} in {full_path}.", f"Price: {price_desc}."]
    if amenities: parts.append(f"Amenities: {amenities}.")

    return ' '.join(parts)