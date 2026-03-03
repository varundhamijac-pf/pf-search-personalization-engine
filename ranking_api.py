"""
ranking_api.py — PF Ranking Model-as-a-Service 
"""

import os
import asyncio
import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Response
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

from pf_features import preprocess_for_model, MODEL_FEATURES

load_dotenv()

logger = logging.getLogger("ranking_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_brain_ref = None
_start_time = time.time()
_request_count = 0

async def load_model():
    global _brain_ref
    path = os.getenv("BRAIN_PATH", "artifacts/brain.pkl")
    if os.path.exists(path):
        
        try:
            new_brain = await asyncio.to_thread(joblib.load, path)
        except Exception as e:
            logger.error(f"Failed to deserialize brain.pkl: {e}")
            return
            
        if not isinstance(new_brain, dict):
            logger.error(f"brain.pkl is not a dict (got {type(new_brain).__name__}). Brain NOT loaded.")
            return

        version = new_brain.get('version', 'unknown')
        stored_features = new_brain.get('features', [])
        if stored_features != MODEL_FEATURES:
            logger.error(
                f"FEATURE MISMATCH: brain.pkl has {stored_features}, "
                f"expected {MODEL_FEATURES}. Brain NOT loaded."
            )
            return
        _brain_ref = new_brain
        logger.info(f"BRAIN loaded: version={version}, features={len(stored_features)}")
    else:
        logger.error(
            f"brain.pkl not found at {path}. "
            f"Ranking service will return HTTP 503 until brain is provided."
        )

@asynccontextmanager
async def lifespan(app):
    await load_model()
    yield

app = FastAPI(title="PF-Ranking-API", version="15.1-MaaS", lifespan=lifespan)

@app.post("/admin/reload-brain")
async def reload_brain(background_tasks: BackgroundTasks):
    background_tasks.add_task(load_model)
    return {"status": "reload_triggered"}

@app.get('/health')
async def health(response: Response):
    if _brain_ref is None:
        response.status_code = 503
        return {
            'status': 'degraded',
            'brain_loaded': False,
            'reason': 'brain.pkl not loaded — check BRAIN_PATH and artifacts volume',
        }
    return {
        'status': 'ok',
        'brain_loaded': True,
        'brain_version': _brain_ref.get('version'),
        'feature_count': len(_brain_ref.get('features', [])),
        'uptime_seconds': round(time.time() - _start_time),
        'total_requests': _request_count, # Note: Per-worker stat in multiprocess environment
    }

class RankRequest(BaseModel):
    properties: List[Dict[str, Any]]

def _predict_sync(properties: list) -> list:
    if not _brain_ref:
        return []

    try:
        df = pd.DataFrame(properties)
    except Exception as e:
        logger.error(
            f"DataFrame construction failed: {e} | "
            f"Input shape: {len(properties)} items, "
            f"sample keys: {list(properties[0].keys()) if properties else 'empty'}"
        )
        return []

    try:
        df = preprocess_for_model(df)
        X = df.reindex(columns=MODEL_FEATURES, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        df['rank_score'] = _brain_ref['model'].predict(X)
    except Exception as e:
        logger.error(f"XGBoost prediction failed: {type(e).__name__}: {e}")
        return [
            {
                "property_listing_id": str(row.get('property_listing_id', '')),
                "rank_score": float(row.get('quality_score', 0.0)),
            }
            for row in properties
        ]

    return [
        {
            "property_listing_id": str(row.get('property_listing_id', '')),
            "rank_score": float(row.get('rank_score', 0.0)),
        }
        for _, row in df.iterrows()
    ]

@app.post("/rank")
async def rank_properties(req: RankRequest):
    global _request_count
    _request_count += 1

    if not _brain_ref:
        logger.warning("POST /rank called but brain is not loaded — returning empty results")
        return {"ranked_results": []}

    if not req.properties:
        return {"ranked_results": []}

    results = await asyncio.to_thread(_predict_sync, req.properties)
    return {"ranked_results": results}