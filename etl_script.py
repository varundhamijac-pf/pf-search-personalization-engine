import pandas as pd
import os
from sqlalchemy import create_engine 
# NOTE: If using Snowflake, import snowflake.connector instead

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
# Update these with your Production DB details
DB_CONFIG = {
    "host": "your-db-host.redshift.amazonaws.com",
    "port": "5439",
    "user": "your_username",
    "password": "your_password",
    "database": "pf_de_prod_db"
}

OUTPUT_FILE = 'listings.parquet'
TRAINING_DATA_FILE = 'user_interactions.csv' # For the monthly training loop

# ==========================================
# 2. SQL QUERIES (From your uploaded files)
# ==========================================

# Source: Listing_Query.txt [cite: 1, 2, 3, 5]
QUERY_INVENTORY = """
WITH 
active_listings AS (
    SELECT 
        property_listing_id, agent_id, key_location, property_type, bedrooms, bathrooms, 
        property_sqft, completion_status, offering_type, furnished_flag, start_time, 
        property_serp_score, pending_verified_flag,
        property_title, property_address
    FROM pf_de_prod_db.pf_dwh.dim_property_listing
    WHERE property_listing_status = 'online'
      AND start_time >= DATEADD(month, -6, GETDATE()) 
),
valid_prices AS (
    SELECT web_id, pp_price FROM pf_de_prod_db.pf_dwh.fct_pricing_all_listings_reporting
    WHERE pp_price > 0
),
geo_data AS (
    SELECT 
        key_location, location_id, coordinates_lat, coordinates_lon,
        COALESCE(location_name_english, location_name_primary) as location_name,
        COALESCE(location_tower_name, location_path_name_primary) as full_location_path
    FROM pf_de_prod_db.pf_dwh.dim_location
    WHERE coordinates_lat IS NOT NULL AND coordinates_lat != 0
)
SELECT 
    l.property_listing_id,
    g.coordinates_lat as latitude,
    g.coordinates_lon as longitude,
    g.location_name,
    l.property_title,
    l.property_type,
    l.bedrooms as beds_int,
    l.bathrooms as bath_int,
    l.property_sqft as size_sqft,
    p.pp_price as price,
    l.start_time as listing_date,
    l.property_serp_score as quality_score,
    COALESCE(s.super_agent_score, 0) as trust_score,
    'https://static.shared.propertyfinder.ae/media/images/listing/default.jpg' as image_url, -- Placeholder until image table is joined
    'Property Consultant' as agent_name -- Placeholder until agent dim is joined
FROM active_listings l
JOIN valid_prices p ON l.property_listing_id = p.web_id
JOIN geo_data g ON l.key_location = g.key_location
LEFT JOIN pf_de_prod_db.pf_dwh.agg_ae_new_superagent_score s ON l.agent_id = s.agent_id;
"""

# Source: User_Engagement_Query.txt [cite: 6, 7, 8]
QUERY_STATS = """
SELECT 
    listing_web_id as property_listing_id,
    SUM(CASE WHEN event_name = 'content_view' THEN 1 ELSE 0 END) as view_count,
    SUM(CASE WHEN event_name IN ('lead_click', 'new_projects_lead_click') THEN 1 ELSE 0 END) as lead_click_count,
    SUM(CASE WHEN event_name IN ('lead_send', 'new_projects_lead_send', 'instapage_lead') THEN 1 ELSE 0 END) as lead_submission_count,
    (
      (SUM(CASE WHEN event_name = 'content_view' THEN 1 ELSE 0 END) * 1.0) +
      (SUM(CASE WHEN event_name IN ('lead_click', 'new_projects_lead_click') THEN 1 ELSE 0 END) * 10.0) +
      (SUM(CASE WHEN event_name IN ('lead_send', 'new_projects_lead_send', 'instapage_lead') THEN 1 ELSE 0 END) * 50.0)
    ) as popularity_score
FROM pf_int_consumer_graph.stg_snowplow_events
WHERE derived_timestamp >= DATEADD(day, -90, GETDATE()) 
  AND listing_web_id IS NOT NULL
GROUP BY listing_web_id;
"""

# ==========================================
# 3. ETL LOGIC
# ==========================================
def get_db_engine():
    # Example for Redshift/Postgres. Modify for your specific DB driver.
    conn_str = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(conn_str)

def run_etl():
    print(" Connecting to Data Warehouse...")
    engine = get_db_engine()

    # 1. Fetch Inventory (The Houses)
    print(" Executing Inventory Query...")
    df_inventory = pd.read_sql(QUERY_INVENTORY, engine)
    print(f"    Loaded {len(df_inventory)} active listings.")

    # 2. Fetch Stats (The Popularity)
    print("Executing Engagement Query...")
    df_stats = pd.read_sql(QUERY_STATS, engine)
    print(f"    Loaded engagement stats for {len(df_stats)} listings.")

    # 3. Merge (Join Inventory + Stats)
    # We use a LEFT JOIN because we want all active houses, even if they have 0 views yet.
    print(" Merging Data...")
    df_final = pd.merge(df_inventory, df_stats, on='property_listing_id', how='left')

    # 4. Clean Up (Fill Nulls)
    # If a listing has no stats, fill with 0
    cols_to_fix = ['view_count', 'lead_click_count', 'lead_submission_count', 'popularity_score']
    df_final[cols_to_fix] = df_final[cols_to_fix].fillna(0)
    
    # Ensure ID is a string for the API
    df_final['property_listing_id'] = df_final['property_listing_id'].astype(str)

    # 5. Save to Parquet
    print(f" Saving to {OUTPUT_FILE}...")
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print("ETL Complete. listings.parquet is ready for the API.")

if __name__ == "__main__":
    run_etl()