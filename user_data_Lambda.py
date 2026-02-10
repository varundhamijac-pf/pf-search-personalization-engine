import boto3
import pandas as pd
import time

# --- CONFIGURATION ---
CLUSTER_ID = "your-redshift-cluster-id"
DB_NAME = "pf_de_prod_db"
DB_USER = "your_db_user"
IAM_ROLE = "arn:aws:iam::123456789012:role/MyRedshiftS3WriteRole"
BUCKET = "your-s3-bucket-name"

PATH_LISTINGS = f"s3://{BUCKET}/temp/listings_base/"
PATH_AMENITIES = f"s3://{BUCKET}/temp/amenities_long/"
PATH_FINAL = f"s3://{BUCKET}/inventory/inventory.parquet"

def run_redshift_query(client, sql, s3_path):
    print(f"Running Query -> {s3_path}...")
    # Escape single quotes for the UNLOAD wrapper
    sql_escaped = sql.replace("'", "''")
    query = f"""
        UNLOAD ('{sql_escaped}')
        TO '{s3_path}'
        IAM_ROLE '{IAM_ROLE}'
        FORMAT PARQUET
        CLEANPATH;
    """
    resp = client.execute_statement(ClusterIdentifier=CLUSTER_ID, Database=DB_NAME, DbUser=DB_USER, Sql=query)
    query_id = resp['Id']
    
    while True:
        status_resp = client.describe_statement(Id=query_id)
        status = status_resp['Status']
        if status == 'FINISHED': break
        elif status == 'FAILED': raise Exception(f"Redshift Failed: {status_resp.get('Error')}")
        time.sleep(5)

def lambda_handler(event, context):
    client = boto3.client('redshift-data')
    
    # --- MASTER QUERY: LISTINGS + WORLD CLASS POPULARITY ---
    sql_master_inventory = """
        WITH active_listings AS (
            SELECT 
                l.property_listing_id, l.agent_id, l.key_location,
                l.property_type, l.property_title, l.property_address,
                l.listing_level, l.pending_verified_flag, l.bedrooms, l.bathrooms,
                l.property_sqft, l.completion_status, l.furnished_flag, l.start_time,
                l.property_serp_score,
                CAST(NULLIF(REGEXP_SUBSTR(l.bedrooms, '[0-9]+'), '') AS INT) as bedrooms_int
            FROM pf_de_prod_db.pf_dwh.dim_property_listing l
            WHERE l.property_listing_status = 'online'
              AND l.start_time >= DATEADD(month, -6, GETDATE()) 
        ),
        valid_prices AS (
            SELECT web_id, pp_price, price_type, offering_type
            FROM pf_de_prod_db.pf_dwh.fct_pricing_all_listings_reporting
            WHERE pp_price IS NOT NULL AND pp_price ~ '^[0-9.]+$' AND CAST(pp_price AS DECIMAL(18,2)) > 0
        ),
        -- THE WORLD CLASS POPULARITY CTE
        listing_popularity AS (
            SELECT 
                listing_web_id,
                SUM(
                    ((CASE WHEN event_name = 'content_view' THEN 1.0 ELSE 0 END) +
                     (CASE WHEN event_name = 'content_save' THEN 5.0 ELSE 0 END) +
                     (CASE WHEN event_name IN ('lead_click', 'lead_send', 'instapage_lead') THEN 15.0 ELSE 0 END))
                    * POWER(0.98, DATEDIFF(day, derived_timestamp, GETDATE()))
                ) as popularity_score
            FROM pf_int_consumer_graph.stg_snowplow_events
            WHERE derived_timestamp >= DATEADD(day, -90, GETDATE())
            GROUP BY 1
            HAVING popularity_score > 1.0
        ),
        geo_data AS (
            SELECT key_location, location_id, coordinates_lat, coordinates_lon,
                   COALESCE(location_name_english, location_name_primary) as location_name,
                   COALESCE(location_tower_name, location_path_name_primary) as full_location_path
            FROM pf_de_prod_db.pf_dwh.dim_location
            WHERE coordinates_lat IS NOT NULL AND coordinates_lat != 0
        )
        SELECT 
            CAST(l.property_listing_id AS VARCHAR) as property_listing_id,
            g.location_id, g.coordinates_lat as latitude, g.coordinates_lon as longitude,
            g.location_name, g.full_location_path, l.property_title,
            l.bedrooms_int as bedrooms, l.bathrooms, l.property_sqft as size_sqft,
            CASE WHEN LOWER(p.offering_type) LIKE '%sale%' THEN 1 ELSE 2 END as category_id,
            CAST(p.pp_price AS DECIMAL(18,2)) as price, p.price_type as price_period,
            COALESCE(pop.popularity_score, 0) as popularity_score,
            COALESCE(s.super_agent_score, 0) as super_agent_score,
            l.pending_verified_flag as verified, l.listing_level
        FROM active_listings l
        JOIN valid_prices p ON l.property_listing_id = p.web_id
        JOIN geo_data g ON l.key_location = g.key_location
        LEFT JOIN listing_popularity pop ON l.property_listing_id = pop.listing_web_id
        LEFT JOIN pf_de_prod_db.pf_dwh.agg_ae_new_superagent_score s ON l.agent_id = s.agent_id
    """

    # --- EXECUTION ---
    run_redshift_query(client, sql_master_inventory, PATH_LISTINGS)
    
    # Query 2: Amenities (Simplified extraction)
    sql_amenities = "SELECT DISTINCT CAST(listing_entity_id AS VARCHAR) as property_listing_id, amenity_code FROM pf_de_prod_db.pf_dwh.dim_property_amenity WHERE end_time = '9999-12-31 00:00:00'"
    run_redshift_query(client, sql_amenities, PATH_AMENITIES)

    # Load and Fix Types for Python Bridge
    df_listings = pd.read_parquet(PATH_LISTINGS)
    df_amenities = pd.read_parquet(PATH_AMENITIES)
    df_listings['property_listing_id'] = df_listings['property_listing_id'].astype(str).str.strip()
    df_amenities['property_listing_id'] = df_amenities['property_listing_id'].astype(str).str.strip()

    # Aggregate Amenities and Merge
    df_amenities_agg = df_amenities.groupby('property_listing_id')['amenity_code'].apply(lambda x: ','.join(set(x.dropna()))).reset_index(name='amenities')
    df_final = df_listings.merge(df_amenities_agg, on='property_listing_id', how='left').fillna({'amenities': ''})

    df_final.to_parquet(PATH_FINAL, index=False)
    return {"status": "Success", "listings": len(df_final)}