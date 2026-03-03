"""
seed_data.py — OpenSearch Indexer + Redis DNA Loader
"""

import pandas as pd
import json
import os
import sys
import time
import logging
import redis
import uuid
from datetime import datetime
from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OS_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
ALIAS_NAME = os.getenv("OPENSEARCH_INDEX", "pf-inventory-v1")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MAPPING_FILE = os.getenv("MAPPING_FILE", "opensearch_mapping.json")

REDIS_DNA_TTL = 259200  
REDIS_PIPELINE_BATCH = 500
BULK_CHUNK_SIZE = 500

# ═══════════════════════════════════════════════════════════════════════
# AWS CHANGE: Support managed OpenSearch (HTTPS on port 443)
# Local Docker: OPENSEARCH_SSL=false, OPENSEARCH_PORT=9200
# AWS Managed:  OPENSEARCH_SSL=true,  OPENSEARCH_PORT=443
# ═══════════════════════════════════════════════════════════════════════
_os_ssl = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"
_os_port = int(os.getenv("OPENSEARCH_PORT", "443" if _os_ssl else "9200"))

os_client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': _os_port}],
    use_ssl=_os_ssl,
    verify_certs=_os_ssl,
    timeout=60,
)
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

LOCK_KEY = "lock:opensearch_seeding"
# FIX BUG 4: Reduced timeout to 30 mins, and lock returns a UUID to allow safe atomic deletion
LOCK_TIMEOUT = 1800  
LOCK_WAIT_TIMEOUT = 600  

def _acquire_lock() -> str:
    """Block until lock is acquired, with timeout. Returns a unique lock value."""
    logger.info("Waiting for seeding lock...")
    lock_value = str(uuid.uuid4())
    start = time.time()
    while not redis_client.set(LOCK_KEY, lock_value, nx=True, ex=LOCK_TIMEOUT):
        if time.time() - start > LOCK_WAIT_TIMEOUT:
            raise TimeoutError(
                f"Could not acquire seeding lock within {LOCK_WAIT_TIMEOUT}s. "
                f"Another seed job may be stuck — check Redis key '{LOCK_KEY}'."
            )
        time.sleep(5)
    logger.info("Lock acquired.")
    return lock_value

def _release_lock(lock_value: str):
    """Atomically releases the lock ONLY if we are still the owner."""
    lua_script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    redis_client.eval(lua_script, 1, LOCK_KEY, lock_value)
    logger.info("Lock released.")


def prepare_document(row: dict) -> dict:
    doc = {k: v for k, v in row.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}

    # --- FIX: Convert dates to ISO 8601 for OpenSearch ---
    for k, v in doc.items():
        # Handle native Pandas/Python datetime objects
        if isinstance(v, (pd.Timestamp, datetime)):
            doc[k] = v.isoformat()
        # Handle raw strings formatted as "YYYY-MM-DD HH:MM:SS"
        elif isinstance(v, str) and len(v) >= 19 and v[10] == ' ':
            # Quick validation to ensure it looks like a date before replacing
            if v[:4].isdigit() and v[5:7].isdigit() and v[8:10].isdigit():
                doc[k] = v.replace(" ", "T", 1)
    # -----------------------------------------------------

    raw_amenities = doc.get('amenities', '')
    if isinstance(raw_amenities, str) and raw_amenities.strip():
        doc['amenities'] = [int(x.strip()) for x in raw_amenities.split(',') if x.strip().isdigit()]
    elif isinstance(raw_amenities, list):
        doc['amenities'] = [int(x) for x in raw_amenities if str(x).isdigit()]
    else:
        doc['amenities'] = []

    lat = float(doc.pop('latitude', 0) or 0)
    lon = float(doc.pop('longitude', 0) or 0)
    if lat != 0 and lon != 0:
        doc['location_coordinates'] = {'lat': lat, 'lon': lon}
    doc['latitude'] = lat
    doc['longitude'] = lon

    # FIX: OpenSearch strict_date_optional_time requires 'T' separator
    date_val = doc.get('listing_date')
    if isinstance(date_val, str) and ' ' in date_val:
        doc['listing_date'] = date_val.replace(' ', 'T')

    vec = doc.get('property_vector')
    if vec is not None and hasattr(vec, 'tolist'):
        doc['property_vector'] = vec.tolist()

    return doc

def _load_mapping() -> dict:
    # AWS CHANGE: Look for mapping file in multiple locations
    for path in [MAPPING_FILE, os.path.join(os.path.dirname(__file__), MAPPING_FILE), "/app/opensearch_mapping.json"]:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(f"opensearch_mapping.json not found in any expected location")

def _get_indices_behind_alias() -> list:
    try:
        alias_info = os_client.indices.get_alias(name=ALIAS_NAME)
        return list(alias_info.keys())
    except Exception:
        return []

def seed_opensearch_full(df: pd.DataFrame):
    # FIX: Ensure unique IDs to prevent the "count stays low" bug
    df = df.drop_duplicates(subset=['property_listing_id']).copy()

    lock_value = _acquire_lock()
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_index = f"{ALIAS_NAME}-{timestamp}"
        mapping = _load_mapping()

        logger.info(f"Creating new index: {new_index} ({len(df)} unique docs)")
        os_client.indices.create(index=new_index, body=mapping)

        actions = []
        for _, row in df.iterrows():
            doc = prepare_document(row.to_dict())
            actions.append({
                "_op_type": "index", "_index": new_index,
                "_id": str(doc['property_listing_id']), "_source": doc,
            })

        success, errors = helpers.bulk(os_client, actions, chunk_size=BULK_CHUNK_SIZE, raise_on_error=False)
        logger.info(f"Indexed {success} docs into {new_index} ({len(errors)} errors)")

        if errors:
            for err in errors[:5]:
                logger.warning(f"Bulk index error: {err}")

        if success < len(df) * 0.9:
            logger.error(
                f"Only {success}/{len(df)} docs indexed (>10% failure). "
                f"Aborting alias swap to protect live index."
            )
            os_client.indices.delete(index=new_index)
            return

        old_indices = _get_indices_behind_alias()
        alias_actions = [{"add": {"index": new_index, "alias": ALIAS_NAME}}]
        for old_idx in old_indices:
            alias_actions.append({"remove": {"index": old_idx, "alias": ALIAS_NAME}})

        try:
            os_client.indices.update_aliases(body={"actions": alias_actions})
        except Exception:
            os_client.indices.put_alias(index=new_index, name=ALIAS_NAME)

        for old_idx in old_indices:
            if old_idx != new_index:
                try:
                    os_client.indices.delete(index=old_idx)
                    logger.info(f"Deleted old index: {old_idx}")
                except Exception:
                    pass

        # SRE FIX: Force refresh so count is immediately accurate for the user
        os_client.indices.refresh(index=ALIAS_NAME)
        logger.info(f"Alias {ALIAS_NAME} now points to {new_index} and is refreshed.")

    finally:
        _release_lock(lock_value)

def seed_opensearch_delta(df: pd.DataFrame):
    lock_value = _acquire_lock()
    try:
        if not os_client.indices.exists(index=ALIAS_NAME) and \
           not os_client.indices.exists_alias(name=ALIAS_NAME):
            logger.error(f"Index/alias {ALIAS_NAME} does not exist. Run full seed first.")
            return

        actions = []
        for _, row in df.iterrows():
            doc = prepare_document(row.to_dict())
            actions.append({
                "_op_type": "index", "_index": ALIAS_NAME,
                "_id": str(doc['property_listing_id']), "_source": doc,
            })

        success, errors = helpers.bulk(os_client, actions, chunk_size=BULK_CHUNK_SIZE, raise_on_error=False)
        logger.info(f"Delta indexed {success} docs ({len(errors)} errors)")

        if errors:
            for err in errors[:5]:
                logger.warning(f"Delta bulk error: {err}")
    finally:
        _release_lock(lock_value)

def seed_redis():
    dna_path = os.path.join(ARTIFACTS_DIR, "user_vectors.parquet")
    if not os.path.exists(dna_path):
        logger.info("No user_vectors.parquet found — skipping Redis seed")
        return

    df = pd.read_parquet(dna_path)
    count = 0
    pipe = redis_client.pipeline()

    # FIX H-1: Try/Except prevents the whole script crashing if one row has bad data
    for _, row in df.iterrows():
        try:
            v1 = row['vector_1']
            v2 = row['vector_2']
            payload = json.dumps({
                'vector_1': v1 if isinstance(v1, list) else v1.tolist(),
                'vector_2': v2 if isinstance(v2, list) else v2.tolist(),
            })
            pipe.set(f"user_dna:{row['user_id']}", payload, ex=REDIS_DNA_TTL)
            count += 1
            if count % REDIS_PIPELINE_BATCH == 0:
                pipe.execute()
                pipe = redis_client.pipeline()
        except Exception as e:
            logger.warning(f"Skipping user {row.get('user_id', '?')} due to error: {e}")
            continue

    pipe.execute()
    logger.info(f"Loaded {count} user DNA profiles into Redis (TTL={REDIS_DNA_TTL}s)")

if __name__ == "__main__":
    is_delta = "--delta" in sys.argv

    if is_delta:
        inv_path = os.path.join(ARTIFACTS_DIR, "inventory_delta.parquet")
        if os.path.exists(inv_path):
            seed_opensearch_delta(pd.read_parquet(inv_path))
        else:
            logger.error(f"Delta inventory not found at {inv_path}")
    else:
        # Priority: inventory.parquet (pipeline output WITH vectors)
        # Fallback: inventory_sample.parquet
        inv_path = os.path.join(ARTIFACTS_DIR, "inventory.parquet")
        if not os.path.exists(inv_path):
            inv_path = os.path.join(ARTIFACTS_DIR, "inventory_sample.parquet")

        if os.path.exists(inv_path):
            df = pd.read_parquet(inv_path)
            if 'property_vector' not in df.columns:
                logger.error(
                    f"FATAL: {inv_path} has NO 'property_vector' column! "
                    f"KNN search will return nothing. Run sagemaker_pipeline.py first."
                )
                sys.exit(1)
            logger.info(f"Loading {inv_path}: {len(df)} rows, vectors ✓")
            seed_opensearch_full(df)
            seed_redis()
        else:
            logger.error(
                f"No inventory file found in {ARTIFACTS_DIR}/. "
                f"Run the pipeline first to generate inventory.parquet with vectors."
            )

    logger.info("Seed complete.")
