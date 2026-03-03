"""
sagemaker_pipeline.py — PF RecSys Training & Embedding Pipeline

"""
from typing import Any
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
import os
import sys
import time
from sklearn.cluster import KMeans

# STRICT MODE: Force sentence-transformers. No fallback logic.
from sentence_transformers import SentenceTransformer
_EMBED_BACKEND = 'sentence_transformers'

from pf_features import (
    MODEL_FEATURES, preprocess_for_model, build_rich_description
)

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Accepts scalars or arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))
BRAIN_VERSION = "v15.3-GeoRanker"  # Added distance_km feature for geo-aware ranking

def _load_embedding_model():
    logger.info(f"Backend: sentence-transformers — {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def generate_vectors(df_listings: pd.DataFrame, model: Any) -> pd.DataFrame:
    logger.info(f"Generating vectors for {len(df_listings)} listings...")

    df_listings = df_listings.copy()
    df_listings['rich_description'] = df_listings.apply(build_rich_description, axis=1)
    texts = df_listings['rich_description'].tolist()

    # Log a sample description so you can verify it looks right
    if texts:
        logger.info(f"Sample rich_description: {texts[0]}")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    df_listings['property_vector'] = list(embeddings)

    logger.info(f"Vectors generated: dim={embeddings.shape[1]}, count={len(embeddings)}")
    return df_listings

def compute_user_dna(df_listings: pd.DataFrame, df_interactions: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing dual-intent User DNA vectors...")

    history = pd.merge(
        df_interactions,
        df_listings[['property_listing_id', 'property_vector']],
        on='property_listing_id',
        how='inner',
    )

    logger.info(f"User DNA merge: {len(df_interactions)} interactions × {len(df_listings)} listings → {len(history)} matched rows")

    if history.empty:
        logger.warning("No interaction-listing overlap. Empty DNA.")
        return pd.DataFrame()

    def _dual_intent(group):
        vectors = np.array(group['property_vector'].tolist())
        weights = (
            np.array(group['interaction_score'].values)
            if 'interaction_score' in group.columns
            else np.ones(len(vectors))
        )
        
        # FIX: Ensure weights are positive (negative weights crash KMeans)
        weights = np.clip(weights, 0.01, None)

        if len(vectors) < 3:
            avg = np.average(vectors, axis=0, weights=weights).tolist()
            return pd.Series({'vector_1': avg, 'vector_2': avg})

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(vectors, sample_weight=weights)
        centers = kmeans.cluster_centers_
        _, counts = np.unique(kmeans.labels_, return_counts=True)
        order = np.argsort(-counts) 

        return pd.Series({
            'vector_1': centers[order[0]].tolist(),
            'vector_2': centers[order[1]].tolist() if len(order) > 1 else centers[order[0]].tolist(),
        })

    user_dna = history.groupby('user_id').apply(_dual_intent).reset_index()
    logger.info(f"Computed DNA for {len(user_dna)} users")
    return user_dna

def train_ranker(df_listings: pd.DataFrame, df_interactions: pd.DataFrame):
    logger.info("Training XGBRanker (rank:ndcg)...")

    # ══════════════════════════════════════════════════════════════════
    # FIX RC-3: Force BOTH sides to str before merge to prevent
    # silent 0-row merges from int64 vs str type mismatch
    # ══════════════════════════════════════════════════════════════════
    df_listings = df_listings.copy()
    df_interactions = df_interactions.copy()
    df_listings['property_listing_id'] = df_listings['property_listing_id'].astype(str).str.strip()
    df_interactions['property_listing_id'] = df_interactions['property_listing_id'].astype(str).str.strip()

    # Diagnostic: Check overlap BEFORE merge
    listing_ids = set(df_listings['property_listing_id'])
    interaction_ids = set(df_interactions['property_listing_id'])
    overlap = listing_ids & interaction_ids
    logger.info(
        f"  Pre-merge check: {len(listing_ids)} listing IDs, "
        f"{len(interaction_ids)} interaction IDs, {len(overlap)} overlapping"
    )
    
    if len(overlap) == 0:
        logger.error(
            "FATAL: Zero overlapping property_listing_id values between listings and interactions! "
            "The model CANNOT train. Check that both CSVs use the same ID format."
        )
        logger.error(f"  Sample listing IDs: {list(listing_ids)[:5]}")
        logger.error(f"  Sample interaction IDs: {list(interaction_ids)[:5]}")
        return

    df_train = pd.merge(df_interactions, df_listings, on='property_listing_id', how='inner')
    
    # ══════════════════════════════════════════════════════════════════
    # FIX CF-A: Diagnostic logging — the merge is the most fragile step
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"  Merged training set: {len(df_train)} rows from {df_train['user_id'].nunique()} users")
    
    if df_train.empty:
        logger.warning("No overlap data for training. Skipping.")
        return

    if len(df_train) < 10:
        logger.warning(
            f"Only {len(df_train)} training rows — model will be weak. "
            f"Consider using more data or checking ID alignment."
        )

    df_train = df_train.sort_values('user_id')

    # ══════════════════════════════════════════════════════════════════
    # NEW: Compute distance_km for training pairs
    #
    # For each user, find their "anchor" listing (highest interaction_score).
    # Then compute haversine distance from that anchor to every other
    # listing the user interacted with. This teaches XGBRanker to weigh
    # geographic proximity as a ranking signal.
    # ══════════════════════════════════════════════════════════════════
    lat_col = 'latitude'
    lon_col = 'longitude'
    has_geo = lat_col in df_train.columns and lon_col in df_train.columns

    if has_geo:
        df_train[lat_col] = pd.to_numeric(df_train[lat_col], errors='coerce').fillna(0.0)
        df_train[lon_col] = pd.to_numeric(df_train[lon_col], errors='coerce').fillna(0.0)

        # Find each user's anchor listing (highest interaction_score)
        score_col = 'interaction_score'
        if score_col not in df_train.columns:
            score_col = None

        if score_col:
            anchor_idx = df_train.groupby('user_id')[score_col].idxmax()
        else:
            # Fallback: first listing per user
            anchor_idx = df_train.groupby('user_id').apply(lambda g: g.index[0])

        anchor_coords = df_train.loc[anchor_idx, ['user_id', lat_col, lon_col]].rename(
            columns={lat_col: 'anchor_lat', lon_col: 'anchor_lon'}
        )
        df_train = df_train.merge(anchor_coords, on='user_id', how='left')

        # Vectorized haversine
        valid_mask = (
            (df_train['anchor_lat'] != 0) & (df_train['anchor_lon'] != 0) &
            (df_train[lat_col] != 0) & (df_train[lon_col] != 0)
        )
        df_train['distance_km'] = 0.0
        if valid_mask.any():
            df_train.loc[valid_mask, 'distance_km'] = haversine_km(
                df_train.loc[valid_mask, 'anchor_lat'].values,
                df_train.loc[valid_mask, 'anchor_lon'].values,
                df_train.loc[valid_mask, lat_col].values,
                df_train.loc[valid_mask, lon_col].values,
            )

        # Log diagnostics
        dist_vals = df_train['distance_km']
        non_zero_dist = (dist_vals > 0).sum()
        logger.info(
            f"  distance_km computed: min={dist_vals.min():.2f}, max={dist_vals.max():.2f}, "
            f"mean={dist_vals.mean():.2f}, non_zero={non_zero_dist}/{len(dist_vals)}"
        )

        # Cleanup temp columns
        df_train.drop(columns=['anchor_lat', 'anchor_lon'], inplace=True, errors='ignore')
    else:
        logger.warning(
            "  ⚠ No latitude/longitude columns in training data — distance_km will be 0.0. "
            "Ensure Redshift ETL exports coordinates."
        )
        df_train['distance_km'] = 0.0

    df_train = preprocess_for_model(df_train, verbose=True)

    X = df_train[MODEL_FEATURES]
    y_float = pd.to_numeric(df_train.get('interaction_score', pd.Series(dtype=float)), errors='coerce').fillna(0)
    
    # ══════════════════════════════════════════════════════════════════
    # FIX RC-2: Scale floats by 10 and round to PRESERVE VARIANCE
    # 
    # Before: ceil(0.13)=1, ceil(0.52)=1, ceil(1.0)=1 → all same label
    # After:  round(0.13*10)=1, round(0.52*10)=5, round(1.0*10)=10 → variance!
    # ══════════════════════════════════════════════════════════════════
    y = np.clip(np.round(y_float * 10).astype(int), 0, 31)
    
    # ══════════════════════════════════════════════════════════════════
    # FIX CF-A: Log label distribution — this is where you catch "Dead Matrix"
    # ══════════════════════════════════════════════════════════════════
    unique_labels = np.unique(y)
    logger.info(
        f"  Label distribution: min={y.min()}, max={y.max()}, "
        f"mean={y.mean():.2f}, unique={len(unique_labels)}, "
        f"values={unique_labels[:10].tolist()}"
    )
    logger.info(
        f"  Raw interaction_score stats: min={y_float.min():.4f}, "
        f"max={y_float.max():.4f}, mean={y_float.mean():.4f}"
    )
    
    # Feature matrix sanity check
    feature_sums = X.sum()
    zero_features = feature_sums[feature_sums == 0].index.tolist()
    if zero_features:
        logger.warning(f"  ⚠ ALL-ZERO features detected: {zero_features}")
    
    non_zero_pct = (X != 0).mean()
    logger.info(f"  Feature fill rates:\n{non_zero_pct.to_string()}")

    if len(unique_labels) < 2:
        logger.error(
            f"FATAL: Only {len(unique_labels)} unique label(s): {unique_labels.tolist()}. "
            f"XGBRanker needs ≥2 distinct labels to learn ranking. "
            f"Check interaction_score values in your data."
        )
        # Still train (it won't crash), but the model will be useless
    
    qid = df_train['user_id'].factorize()[0]

    # FIX: ndcg_exp_gain=False for XGBoost ≥ 2.1 compatibility (labels must be ≤ 31 with exp gain)
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=300,
        learning_rate=0.02,
        max_depth=6,
        tree_method='hist',
        random_state=42,
        ndcg_exp_gain=False,
    )

    if len(X) > 0:
        model.fit(X, y, qid=qid)
        logger.info("XGBRanker trained successfully.")
        
        # ══════════════════════════════════════════════════════════════
        # FIX CF-B: Post-training validation — predict on training data
        # to confirm model actually learned something
        # ══════════════════════════════════════════════════════════════
        sample_preds = model.predict(X.head(min(10, len(X))))
        logger.info(
            f"  Post-training validation scores (first {len(sample_preds)}): "
            f"{[round(float(s), 4) for s in sample_preds]}"
        )
        if all(s == 0.0 for s in sample_preds):
            logger.error(
                "⚠ BRAINDEAD MODEL: All validation predictions are 0.0! "
                "The model learned nothing. Check label variance and feature matrix."
            )
        else:
            pred_range = float(sample_preds.max()) - float(sample_preds.min())
            logger.info(f"  Prediction range: {pred_range:.4f} (>0 means model is alive)")
    else:
        logger.warning("Empty training set — model not fitted.")

    brain = {
        "version": BRAIN_VERSION,
        "model": model,
        "features": MODEL_FEATURES,
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    brain_path = os.path.join(ARTIFACTS_DIR, "brain.pkl")
    joblib.dump(brain, brain_path)
    logger.info(f"Saved brain.pkl: version={BRAIN_VERSION}, features={len(MODEL_FEATURES)}")

def export_artifacts(df_listings: pd.DataFrame, user_dna: pd.DataFrame, is_delta: bool = False):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    drop_cols = ['rich_description', 'rent_multiplier']
    df_out = df_listings.drop(columns=[c for c in drop_cols if c in df_listings.columns], errors='ignore')

    suffix = "_delta" if is_delta else ""
    inv_path = os.path.join(ARTIFACTS_DIR, f"inventory{suffix}.parquet")
    df_out.to_parquet(inv_path, index=False)
    logger.info(f"Saved {inv_path}: {len(df_out)} listings")

    if not is_delta and not user_dna.empty:
        dna_path = os.path.join(ARTIFACTS_DIR, "user_vectors.parquet")
        user_dna.to_parquet(dna_path, index=False)
        logger.info(f"Saved {dna_path}: {len(user_dna)} user profiles")

    return df_out

def run_full_pipeline(inventory_path: str, interactions_path: str):
    t0 = time.time()
    logger.info("=== FULL PIPELINE START ===")

    df_listings = pd.read_parquet(inventory_path)
    df_interactions = pd.read_parquet(interactions_path)
    logger.info(f"Loaded {len(df_listings)} listings, {len(df_interactions)} interactions")
    
    # ══════════════════════════════════════════════════════════════════
    # FIX RC-3: Normalize IDs immediately after loading
    # ══════════════════════════════════════════════════════════════════
    df_listings['property_listing_id'] = df_listings['property_listing_id'].astype(str).str.strip()
    df_interactions['property_listing_id'] = df_interactions['property_listing_id'].astype(str).str.strip()
    df_interactions['user_id'] = df_interactions['user_id'].astype(str).str.strip()
    
    # Log data quality
    logger.info(f"  Listing ID dtype: {df_listings['property_listing_id'].dtype}, sample: {df_listings['property_listing_id'].head(3).tolist()}")
    logger.info(f"  Interaction ID dtype: {df_interactions['property_listing_id'].dtype}, sample: {df_interactions['property_listing_id'].head(3).tolist()}")
    
    if 'interaction_score' in df_interactions.columns:
        scores = df_interactions['interaction_score']
        logger.info(
            f"  Interaction scores: min={scores.min():.4f}, max={scores.max():.4f}, "
            f"mean={scores.mean():.4f}, zeros={int((scores == 0).sum())}/{len(scores)}"
        )

    model = _load_embedding_model()
    df_listings = generate_vectors(df_listings, model)

    user_dna = compute_user_dna(df_listings, df_interactions)

    train_ranker(df_listings, df_interactions)

    df_out = export_artifacts(df_listings, user_dna, is_delta=False)

    try:
        from seed_data import seed_opensearch_full, seed_redis
        seed_opensearch_full(df_out)
        seed_redis()
        logger.info("OpenSearch index swapped and Redis DNA loaded.")
    except Exception as e:
        logger.error(f"Seeding failed (manual seed_data.py run required): {e}")

    elapsed = time.time() - t0
    logger.info(f"=== FULL PIPELINE COMPLETE in {elapsed:.1f}s ===")

def run_delta_pipeline(delta_inventory_path: str):
    t0 = time.time()
    logger.info("=== DELTA PIPELINE START ===")

    df_delta = pd.read_parquet(delta_inventory_path)
    logger.info(f"Loaded {len(df_delta)} delta listings")

    if df_delta.empty:
        logger.info("No new listings — skipping delta.")
        return

    model = _load_embedding_model()
    df_delta = generate_vectors(df_delta, model)

    df_out = export_artifacts(df_delta, pd.DataFrame(), is_delta=True)

    try:
        from seed_data import seed_opensearch_delta
        seed_opensearch_delta(df_out)
        logger.info(f"Delta upserted {len(df_out)} new listings into OpenSearch.")
    except Exception as e:
        logger.error(f"Delta upsert failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"=== DELTA PIPELINE COMPLETE in {elapsed:.1f}s ===")

if __name__ == "__main__":
    is_delta = "--delta" in sys.argv

    if is_delta:
        inv_path = (
            sys.argv[sys.argv.index("--delta") + 1]
            if len(sys.argv) > sys.argv.index("--delta") + 1
            else os.path.join(ARTIFACTS_DIR, "inventory_delta.parquet")
        )
        run_delta_pipeline(inv_path)
    else:
        inv = os.path.join(ARTIFACTS_DIR, "inventory.parquet")
        inter = os.path.join(ARTIFACTS_DIR, "interactions.parquet")
        run_full_pipeline(inv, inter)