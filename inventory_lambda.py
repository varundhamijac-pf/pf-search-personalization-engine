import boto3
import pandas as pd
import time

# --- CONFIGURATION ---
# DevOps: Update these values for the production environment
CLUSTER_ID = "your-redshift-cluster-id"
DB_NAME = "pf_de_prod_db"
DB_USER = "your_db_user"
IAM_ROLE = "arn:aws:iam::123456789012:role/MyRedshiftS3WriteRole"
BUCKET = "your-s3-bucket-name"

# S3 Output Paths
PATH_LISTINGS = f"s3://{BUCKET}/temp/listings_base/"
PATH_AMENITIES = f"s3://{BUCKET}/temp/amenities_long/"
PATH_FINAL = f"s3://{BUCKET}/inventory/inventory.parquet"

def run_redshift_query(client, sql, s3_path):
    """Executes Redshift UNLOAD and waits for completion."""
    print(f"Running Query -> {s3_path}...")
    # Escape single quotes for Redshift UNLOAD string
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
        if status == 'FINISHED':
            print(f"Query Complete: {s3_path}")
            break
        elif status == 'FAILED':
            err = status_resp.get('Error', 'Unknown Error')
            raise Exception(f"Redshift Failed: {err}")
        time.sleep(5)

def lambda_handler(event, context):
    client = boto3.client('redshift-data')
    
    # --- UPDATED QUERY 1: ADVANCED LISTINGS LOGIC ---
    # Integrated CTEs for price cleaning, location mapping, and bedroom extraction
    sql_listings = """
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
            SELECT 
                web_id, pp_price, price_type, offering_type
            FROM pf_de_prod_db.pf_dwh.fct_pricing_all_listings_reporting
            WHERE pp_price IS NOT NULL 
              AND pp_price ~ '^[0-9.]+$' 
              AND CAST(pp_price AS DECIMAL(18,2)) > 0
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
            CAST(l.property_listing_id AS VARCHAR) as property_listing_id,
            g.location_id, g.coordinates_lat as latitude, g.coordinates_lon as longitude,
            g.location_name, g.full_location_path,
            l.property_title, l.property_address, l.listing_level, l.pending_verified_flag,
            l.start_time as listing_date, l.property_type, l.bedrooms_int as bedrooms,
            l.bedrooms as bedrooms_label, l.bathrooms, l.property_sqft as size_sqft,
            l.completion_status, l.furnished_flag,
            CASE WHEN LOWER(p.offering_type) LIKE '%sale%' THEN 1 WHEN LOWER(p.offering_type) LIKE '%rent%' THEN 2 ELSE 0 END as category_id,
            CASE 
                WHEN LOWER(l.property_type) LIKE '%apartment%' THEN 1
                WHEN LOWER(l.property_type) LIKE '%villa%' THEN 35
                WHEN LOWER(l.property_type) LIKE '%townhouse%' THEN 22
                WHEN LOWER(l.property_type) LIKE '%penthouse%' THEN 20
                ELSE 0 
            END as property_type_id,
            CAST(p.pp_price AS DECIMAL(18,2)) as price, p.price_type as price_period,
            l.property_serp_score as quality_score,
            COALESCE(s.super_agent_score, 0) as super_agent_score
        FROM active_listings l
        JOIN valid_prices p ON l.property_listing_id = p.web_id
        JOIN geo_data g ON l.key_location = g.key_location
        LEFT JOIN pf_de_prod_db.pf_dwh.agg_ae_new_superagent_score s ON l.agent_id = s.agent_id
    """

    # --- QUERY 2: AMENITIES ---
    sql_amenities = """
        SELECT DISTINCT 
            CAST(pa.listing_entity_id AS VARCHAR) as property_listing_id, 
            da.amenity_code
        FROM pf_de_prod_db.pf_dwh.dim_property_amenity pa
        JOIN pf_de_prod_db.pf_dwh.dim_amenity da ON pa.amenity_id = da.amenity_id
        WHERE pa.end_time = '9999-12-31 00:00:00'
          AND pa.meta_event_type != 'D'
    """

    # 1. Execute Redshift Exports
    run_redshift_query(client, sql_listings, PATH_LISTINGS)
    run_redshift_query(client, sql_amenities, PATH_AMENITIES)

    print("Starting Python Merge and Sanitization...")
    
    # 2. Load into Pandas
    df_listings = pd.read_parquet(PATH_LISTINGS, engine='pyarrow')
    df_amenities = pd.read_parquet(PATH_AMENITIES, engine='pyarrow')
    
    # 3. Type-Safe Bridge: Ensure IDs are clean strings
    df_listings['property_listing_id'] = df_listings['property_listing_id'].astype(str).str.strip()
    df_amenities['property_listing_id'] = df_amenities['property_listing_id'].astype(str).str.strip()

    # 4. Aggregate Amenities (Join codes with commas)
    df_amenities_agg = df_amenities.groupby('property_listing_id')['amenity_code'].apply(
        lambda x: ','.join(set(filter(None, x.dropna())))
    ).reset_index(name='amenities')

    # 5. Final Merge
    df_final = pd.merge(df_listings, df_amenities_agg, on='property_listing_id', how='left')
    
    # 6. Data Cleaning
    df_final['amenities'] = df_final['amenities'].fillna('')
    # Ensure binary flags are consistent
    df_final['pending_verified_flag'] = df_final['pending_verified_flag'].fillna(0).astype(int)

    # 7. Save Final Production Inventory
    print(f"Saving {len(df_final)} rows to {PATH_FINAL}...")
    df_final.to_parquet(PATH_FINAL, index=False, engine='pyarrow')
    
    print("ETL Process Complete.")
    return {"statusCode": 200, "body": f"Inventory Updated with {len(df_final)} listings"}