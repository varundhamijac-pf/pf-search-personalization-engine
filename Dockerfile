FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Python deps — full ML stack (for SageMaker pipeline)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY pf_features.py .
COPY ranking_api.py .
COPY search_api.py .
COPY recsys_api.py .
COPY seed_data.py .
COPY sagemaker_pipeline.py .
COPY opensearch_mapping.json .

# Create dirs for SageMaker Processing Job paths
RUN mkdir -p /app/artifacts /opt/ml/processing/input /opt/ml/processing/output

# Default: health check port
EXPOSE 8000 8001 8002
