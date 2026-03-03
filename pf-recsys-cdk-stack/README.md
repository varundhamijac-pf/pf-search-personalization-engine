# PropertyFinder RecSys — AWS CDK Deployment

## Prerequisites

1. **AWS CLI** configured with credentials
2. **Node.js** >= 18 (CDK requires it)
3. **Python** >= 3.11
4. **AWS CDK CLI**: `npm install -g aws-cdk`

## Project Structure

```
cdk-stack/
├── app.py                          # CDK app entry point
├── pf_recsys_stack.py             # Main infrastructure stack
├── cdk.json                        # CDK config
├── requirements.txt                # CDK Python dependencies
├── lambda/
│   ├── etl/
│   │   └── redshift_data.py       # Daily + hourly ETL Lambda
│   └── post-training/
│       └── post_training.py       # Post-SageMaker: seed + reload
└── README.md                       # This file
```

## What Gets Created

| Resource | Service | Details |
|----------|---------|---------|
| VPC | EC2 | 2 AZs, public + private subnets, 1 NAT Gateway |
| S3 Bucket | S3 | Versioned, 30-day lifecycle on gold/ prefix |
| ECR Repos | ECR | pf-recsys-api + pf-recsys-training |
| OpenSearch | OpenSearch Service | 2x r6g.large, 100GB gp3, HNSW ready |
| Redis | ElastiCache | 2x cache.r6g.large, multi-AZ, auto-failover |
| ECS Cluster | Fargate | 5 tasks (1 ranking + 2 recsys + 2 search) |
| ALB | ELB | Path-based routing to 3 services |
| Auto-scaling | App Auto Scaling | CPU 70% + 500 req/target for recsys & search |
| ETL Lambda | Lambda | Redshift → S3, daily + hourly schedules |
| Post-training Lambda | Lambda | Seed OpenSearch/Redis + reload brain |
| SageMaker Role | IAM | For processing job (training) |
| EventBridge Rules | EventBridge | Daily 22:00 UTC, hourly delta |
| Alarms | CloudWatch | Unhealthy hosts, high latency, 5xx, ETL errors |
| Alerts | SNS | Email notifications |

## Deploy

### Step 1: Install dependencies

```bash
cd cdk-stack
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Step 2: Configure

Edit `cdk.json` with your account ID and region:
```json
{
  "context": {
    "account": "123456789012",
    "region": "me-south-1"
  }
}
```

### Step 3: Bootstrap CDK (first time only)

```bash
cdk bootstrap aws://123456789012/me-south-1
```

### Step 4: Deploy

```bash
# Preview changes
cdk diff

# Deploy (will prompt for approval)
cdk deploy --parameters AlertEmail=team@propertyfinder.com \
           --parameters RedshiftClusterId=pf-prod-cluster \
           --parameters RedshiftDB=pf_de_prod_db \
           --parameters RedshiftUser=recsys_service \
           --parameters RedshiftIAMRoleArn=arn:aws:iam::ACCOUNT:role/RedshiftS3Access
```

### Step 5: Build and push Docker images

After CDK creates the ECR repos, push your images:

```bash
# Get ECR login
aws ecr get-login-password --region me-south-1 | \
  docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com

# Build and push API image
docker build -f Dockerfile.api -t pf-recsys-api .
docker tag pf-recsys-api:latest <ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com/pf-recsys-api:latest
docker push <ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com/pf-recsys-api:latest

# Build and push training image
docker build -f Dockerfile -t pf-recsys-training .
docker tag pf-recsys-training:latest <ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com/pf-recsys-training:latest
docker push <ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com/pf-recsys-training:latest

# Force ECS to pull new images
aws ecs update-service --cluster pf-recsys --service RankingService --force-new-deployment
aws ecs update-service --cluster pf-recsys --service RecSysService --force-new-deployment
aws ecs update-service --cluster pf-recsys --service SearchService --force-new-deployment
```

### Step 6: Run first training

```bash
# Trigger ETL
aws lambda invoke --function-name pf-recsys-etl \
  --payload '{"is_hourly_delta": false}' output.json

# Run SageMaker training (use the output SageMakerRoleArn from CDK)
python trigger_sagemaker.py  # See SageMaker section below

# Or manually trigger post-training
aws lambda invoke --function-name pf-recsys-post-training \
  --payload '{}' output.json
```

## SageMaker Training

Run training as a SageMaker Processing Job:

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

processor = ScriptProcessor(
    image_uri='<ACCOUNT>.dkr.ecr.me-south-1.amazonaws.com/pf-recsys-training:latest',
    role='<SageMakerRoleArn from CDK output>',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    command=['python3'],
    max_runtime_in_seconds=7200,
)

processor.run(
    code='sagemaker_pipeline.py',
    inputs=[
        ProcessingInput(
            source='s3://pf-recsys-ACCOUNT-me-south-1/gold/snapshot_date=YYYY-MM-DD/inventory/',
            destination='/opt/ml/processing/input/inventory'
        ),
        ProcessingInput(
            source='s3://pf-recsys-ACCOUNT-me-south-1/gold/snapshot_date=YYYY-MM-DD/interactions/',
            destination='/opt/ml/processing/input/interactions'
        ),
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output/',
            destination='s3://pf-recsys-ACCOUNT-me-south-1/artifacts/'
        ),
    ],
)
```

## Rollback

```bash
# Rollback to previous brain.pkl (S3 versioning)
aws s3api list-object-versions --bucket pf-recsys-ACCOUNT-me-south-1 --prefix artifacts/brain.pkl
aws s3api get-object --bucket pf-recsys-ACCOUNT-me-south-1 --key artifacts/brain.pkl \
  --version-id <previous-version-id> brain.pkl.rollback

# Kill all services (emergency)
aws ecs update-service --cluster pf-recsys --service RankingService --desired-count 0
aws ecs update-service --cluster pf-recsys --service RecSysService --desired-count 0
aws ecs update-service --cluster pf-recsys --service SearchService --desired-count 0

# Destroy everything (WARNING: irreversible for data stores with RETAIN policy)
cdk destroy
```

## Estimated Monthly Cost

| Service | Estimated |
|---------|-----------|
| ECS Fargate (5 tasks) | ~$150 |
| OpenSearch (2x r6g.large) | ~$350 |
| ElastiCache (2x r6g.large) | ~$300 |
| ALB + data transfer | ~$25 |
| NAT Gateway | ~$35 |
| Lambda + EventBridge | ~$5 |
| SageMaker (1hr/day) | ~$10 |
| S3 + CloudWatch | ~$20 |
| **Total** | **~$895/month** |
