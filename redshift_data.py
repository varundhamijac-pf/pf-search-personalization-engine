"""
redshift_data.py — AWS Lambda: Redshift → S3 Parquet ETL
"""

import boto3
import os
import time
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLUSTER_ID = os.getenv("REDSHIFT_CLUSTER_ID", "pf-prod-cluster")
DB_NAME = os.getenv("REDSHIFT_DB", "pf_de_prod_db")
DB_USER = os.getenv("REDSHIFT_USER", "recsys_service")
IAM_ROLE = os.getenv("REDSHIFT_IAM_ROLE")
BUCKET = os.getenv("S3_BUCKET", "pf-recsys-prod")

_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 840  

def lambda_handler(event, context):
    client = boto3.client('redshift-data')
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    snapshot_hour = datetime.now().strftime("%H")
    is_delta = event.get("is_hourly_delta", False)

    if is_delta:
        base_s3_path = f"s3://{BUCKET}/gold/snapshot_date={snapshot_date}/delta_hour={snapshot_hour}"
    else:
        base_s3_path = f"s3://{BUCKET}/gold/snapshot_date={snapshot_date}"

    time_filter = "AND l.start_time >= DATEADD(hour, -2, GETDATE())" if is_delta else ""

    inventory_sql = f"""
        WITH unique_amenities AS (
            -- OPTIMIZATION 1: Use LISTAGG(DISTINCT) to avoid expensive nested subqueries
            SELECT 
                pa.listing_entity_id AS property_listing_id,
                LISTAGG(DISTINCT da.amenity_code, ',') WITHIN GROUP (ORDER BY da.amenity_code) AS amenities
            FROM pf_de_prod_db.pf_dwh.dim_property_amenity pa
            JOIN pf_de_prod_db.pf_dwh.dim_amenity da ON pa.amenity_id = da.amenity_id
            WHERE pa.end_time = '9999-12-31 00:00:00' AND pa.meta_event_type != 'D'
            GROUP BY 1
        ),
        listing_popularity_base AS (
            -- OPTIMIZATION: Filter early, use indexes (no ::DATE in WHERE)
            SELECT
                contexts_ae_propertyfinder_listing_context_1[0].web_id::BIGINT AS property_listing_id,
                event_name,
                DATEDIFF('day', derived_tstamp::DATE, CURRENT_DATE) AS days_ago
            FROM pf_prod_enr.snowplow_transformed.snowplow_transformed_events
            WHERE derived_tstamp >= DATEADD('day', -90, CURRENT_DATE)
              AND contexts_ae_propertyfinder_listing_context_1 IS NOT NULL
              AND event_name IN (
                  'content_view', 'content_save', 
                  'lead_click', 'new_projects_lead_click', 'new_projects_dev_lead_click', 
                  'lead_send', 'new_projects_lead_send', 'new_projects_dev_lead_send'
              )
        ),
        daily_aggregated_popularity AS (
            -- OPTIMIZATION: Pre-aggregate event scores per day BEFORE the heavy math
            SELECT
                property_listing_id,
                days_ago,
                SUM(
                    CASE WHEN event_name = 'content_view' THEN 1.0 
                         WHEN event_name = 'content_save' THEN 5.0 
                         WHEN event_name IN ('lead_click','new_projects_lead_click','new_projects_dev_lead_click') THEN 10.0 
                         WHEN event_name IN ('lead_send','new_projects_lead_send','new_projects_dev_lead_send') THEN 50.0 
                         ELSE 0.0 
                    END
                ) AS daily_raw_score
            FROM listing_popularity_base
            GROUP BY 1, 2
        ),
        listing_popularity AS (
            -- OPTIMIZATION: Perform the POWER() math on the drastically reduced daily rows
            SELECT
                property_listing_id,
                CAST(SUM(daily_raw_score * POWER(0.98, days_ago)) AS DECIMAL(10,4)) AS popularity_score
            FROM daily_aggregated_popularity
            GROUP BY 1
            HAVING popularity_score > 1.0
        ),
        active_listings AS (
            SELECT
                l.property_listing_id, l.agent_id, l.key_location,
                l.property_type, l.property_type_id,
                l.listing_level, l.pending_verified_flag,
                CAST(NULLIF(REGEXP_SUBSTR(l.bedrooms, '[0-9]+'), '') AS INT) AS bedrooms_int,
                l.bedrooms AS bedrooms_label,
                l.bathrooms, l.property_sqft,
                l.completion_status, l.furnished_flag,
                l.start_time, l.property_serp_score
            FROM pf_de_prod_db.pf_dwh.dim_property_listing l
            WHERE l.property_listing_status = 'online'
              AND (
                  l.start_time >= DATEADD('month', -6, CURRENT_DATE)
                  OR (l.completion_status = 'off_plan' AND l.start_time >= DATEADD('month', -24, CURRENT_DATE))
              )
              {time_filter}
        ),
        valid_prices AS (
            -- OPTIMIZATION 2: Replaced the slow regex (~ '^[0-9.]+$') with standard string checks
            SELECT web_id, p.pp_price, p.price_type, p.offering_type
            FROM pf_de_prod_db.pf_dwh.fct_pricing_all_listings_reporting p
            WHERE p.pp_price IS NOT NULL
              AND LEN(TRIM(p.pp_price)) > 0
              AND p.pp_price NOT LIKE '%[^0-9.]%'
              AND CAST(p.pp_price AS DECIMAL(18,2)) > 0
        ),
        geo_data AS (
            SELECT
                key_location, location_id,
                coordinates_lat, coordinates_lon,
                COALESCE(location_name_english, location_name_primary) AS location_name,
                COALESCE(location_tower_name, location_path_name_primary) AS full_location_path
            FROM pf_de_prod_db.pf_dwh.dim_location
            WHERE coordinates_lat IS NOT NULL AND coordinates_lat != 0
        )
        SELECT
            l.property_listing_id,
            g.location_id,
            g.coordinates_lat AS latitude,
            g.coordinates_lon AS longitude,
            g.location_name,
            g.full_location_path,
            l.listing_level,
            l.pending_verified_flag AS is_verified,
            l.start_time AS listing_date,
            l.property_type,
            l.property_type_id,
            l.bedrooms_int AS bedrooms,
            l.bedrooms_label,
            l.bathrooms,
            l.property_sqft AS size_sqft,
            l.completion_status,
            l.furnished_flag,
            CASE
                WHEN p.offering_type = 'Residential for Sale' THEN 1
                WHEN p.offering_type = 'Residential for Rent' THEN 2
                WHEN p.offering_type = 'Commercial for Sale' THEN 3
                WHEN p.offering_type = 'Commercial for Rent' THEN 4
                ELSE 0
            END AS category_id,
            CAST(p.pp_price AS DECIMAL(18,2)) AS price,
            p.price_type AS price_period,
            l.property_serp_score AS quality_score,
            COALESCE(s.super_agent_score, 0) AS super_agent_score,
            COALESCE(a.amenities, '') AS amenities,
            COALESCE(pop.popularity_score, 0.0) AS popularity_score,
            l.agent_id
        FROM active_listings l
        JOIN valid_prices p ON l.property_listing_id = p.web_id
        JOIN geo_data g ON l.key_location = g.key_location
        LEFT JOIN pf_de_prod_db.pf_dwh.dim_agent ag ON l.agent_id = ag.agent_id
        LEFT JOIN pf_de_prod_db.pf_dwh.agg_ae_new_superagent_score s ON l.agent_id = s.agent_id
        LEFT JOIN unique_amenities a ON l.property_listing_id = a.property_listing_id
        LEFT JOIN listing_popularity pop ON l.property_listing_id = pop.property_listing_id;
    """

    inventory_s3 = f"{base_s3_path}/inventory/"
    _execute_unload(client, inventory_sql, inventory_s3)

    if not is_delta:
        interactions_sql = """
            WITH base_events AS (
                -- Filter out everything except the last 3 months right at the source
                SELECT
                    COALESCE(user_id, domain_userid) AS user_id,
                    contexts_ae_propertyfinder_listing_context_1[0].web_id::BIGINT AS property_listing_id,
                    event_name,
                    DATEDIFF('day', derived_tstamp::DATE, GETDATE()::DATE) AS days_ago
                FROM pf_prod_enr.snowplow_transformed.snowplow_transformed_events
                -- CHANGED: -12 months is now -3 months
                WHERE derived_tstamp >= DATEADD('month', -3, CURRENT_DATE)
                  AND contexts_ae_propertyfinder_listing_context_1 IS NOT NULL
                  AND event_name IN (
                      'content_view', 'content_save', 'content_unsave', 
                      'lead_click', 'new_projects_lead_click', 'new_projects_dev_lead_click', 
                      'lead_send', 'new_projects_lead_send', 'new_projects_dev_lead_send', 
                      'instapage_lead', 'leadsbridge_lead'
                  )
            ),
            daily_aggregated_events AS (
                -- Pre-aggregate by day to save calculation time
                SELECT
                    user_id,
                    property_listing_id,
                    days_ago,
                    SUM(
                        CASE WHEN event_name = 'content_view' THEN 1.0 
                             WHEN event_name = 'content_save' THEN 5.0 
                             WHEN event_name = 'content_unsave' THEN -5.0 
                             WHEN event_name IN ('lead_click','new_projects_lead_click','new_projects_dev_lead_click') THEN 10.0 
                             WHEN event_name IN ('lead_send','new_projects_lead_send','new_projects_dev_lead_send','instapage_lead','leadsbridge_lead') THEN 50.0 
                             ELSE 0.0 
                        END
                    ) AS daily_raw_score
                FROM base_events
                GROUP BY 1, 2, 3
            ),
            enriched_events AS (
                SELECT
                    e.user_id,
                    e.property_listing_id,
                    e.days_ago,
                    e.daily_raw_score,
                    p.offering_type,
                    CAST(p.pp_price AS DECIMAL(18,2)) AS price
                FROM daily_aggregated_events e
                JOIN pf_de_prod_db.pf_dwh.fct_pricing_all_listings_reporting p 
                  ON e.property_listing_id = p.web_id
                -- CHANGED: Since we only have 3 months of data, cap both rent/sale at 90 days
                WHERE e.days_ago <= 90
            )
            SELECT
                user_id,
                property_listing_id,
                SUM(daily_raw_score * POWER(CASE WHEN price > 1000000 THEN 0.98 ELSE 0.95 END, days_ago)) AS interaction_score
            FROM enriched_events
            GROUP BY 1, 2
            HAVING interaction_score != 0;
        """
        interactions_s3 = f"{base_s3_path}/interactions/"
        _execute_unload(client, interactions_sql, interactions_s3)

    logger.info(f"ETL complete → {base_s3_path} (delta={is_delta})")
    return {"status": "success", "s3_base_path": base_s3_path, "is_delta": is_delta}

def _execute_unload(client, sql: str, s3_path: str):
    unload_sql = (
        f"UNLOAD ($${sql}$$) "
        f"TO '{s3_path}' "
        f"IAM_ROLE '{IAM_ROLE}' "
        f"FORMAT PARQUET CLEANPATH PARALLEL ON;"
    )

    response = client.execute_statement(
        ClusterIdentifier=CLUSTER_ID, Database=DB_NAME, DbUser=DB_USER, Sql=unload_sql,
    )
    stmt_id = response['Id']
    
    elapsed = 0
    while elapsed < _POLL_TIMEOUT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS

        desc = client.describe_statement(Id=stmt_id)
        status = desc['Status']

        if status == 'FINISHED':
            return

        if status in ('FAILED', 'ABORTED'):
            raise RuntimeError(f"Redshift UNLOAD failed: {desc.get('Error', 'unknown')}")

    raise TimeoutError(f"Redshift UNLOAD hit {_POLL_TIMEOUT_SECONDS}s safety timeout (stmt_id={stmt_id})")