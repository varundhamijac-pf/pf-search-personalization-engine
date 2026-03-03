"""
post_training.py — Lambda triggered after SageMaker training completes.

Steps:
  1. Downloads artifacts from S3 (brain.pkl, inventory.parquet, user_vectors.parquet)
  2. Seeds OpenSearch (blue-green index swap)
  3. Loads User DNA into Redis
  4. Hot-reloads ranking API brain
"""

import os
import json
import boto3
import logging
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.getenv("S3_BUCKET", "pf-recsys-prod")
RANKING_API_URL = os.getenv("RANKING_API_URL", "http://ranking-api.pf-recsys.local:8002")
ARTIFACTS_DIR = "/tmp/artifacts"


def lambda_handler(event, context):
    s3 = boto3.client("s3")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Step 1: Download artifacts from S3
    artifacts = ["brain.pkl", "inventory.parquet", "user_vectors.parquet"]
    for artifact in artifacts:
        s3_key = f"artifacts/{artifact}"
        local_path = os.path.join(ARTIFACTS_DIR, artifact)
        try:
            s3.download_file(S3_BUCKET, s3_key, local_path)
            logger.info(f"Downloaded s3://{S3_BUCKET}/{s3_key} → {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            if artifact == "brain.pkl":
                return {"status": "error", "reason": f"brain.pkl download failed: {e}"}

    # Step 2: Seed OpenSearch + Redis
    # Import seed_data (packaged in the Lambda layer or bundled)
    try:
        os.environ["ARTIFACTS_DIR"] = ARTIFACTS_DIR
        os.environ["OPENSEARCH_SSL"] = "true"
        os.environ["OPENSEARCH_PORT"] = "443"

        # These are imported at runtime to use the env vars set above
        from seed_data import seed_opensearch_full, seed_redis
        import pandas as pd

        inv_path = os.path.join(ARTIFACTS_DIR, "inventory.parquet")
        if os.path.exists(inv_path):
            df = pd.read_parquet(inv_path)
            logger.info(f"Seeding OpenSearch with {len(df)} listings...")
            seed_opensearch_full(df)
            logger.info("OpenSearch seeded successfully.")

        seed_redis()
        logger.info("Redis DNA loaded successfully.")
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        # Continue to brain reload — seeding failure is not fatal

    # Step 3: Hot reload ranking API brain
    try:
        req = urllib.request.Request(
            f"{RANKING_API_URL}/admin/reload-brain",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            logger.info(f"Brain reload triggered: {result}")
    except Exception as e:
        logger.error(f"Brain reload failed: {e}")

    # Step 4: Verify health
    try:
        with urllib.request.urlopen(f"{RANKING_API_URL}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode())
            logger.info(f"Ranking API health: {health}")
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    return {"status": "success"}
