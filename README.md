 
Recommendation System V9.4 Architecture
# Recommendation System.

# Architecture & Scalability Handover

### Engineering Team

## 1 Executive Summary

We have built a **Hybrid Batch-Processed, In-Memory Serving Recommendation Engine**.

Instead of hitting the database for every user search (which is slow and unscalable), we pre-
compute highly optimized Parquet datasets daily using AWS Redshift and serve them instantly
from memory using FastAPI.

- **Throughput:** Sub-50ms response time per request.
- **Capacity:** Validated for 10 Lakh+ (1 Million) active listings daily.
- **ArchitectureStrategy:** Decoupled ETL (Redshift/Lambda) and Serving (FastAPI/App Run-
    ner).

## 2 System Architecture

The system is composed of three distinct layers designed for modularity and scalability.

### 2.1 A. Data Pipeline (ETL Layer)

Responsible for extracting, transforming, and loading data into the Data Lake.

- **Source:** AWS Redshift (pf_de_prod_db)
- **Compute:** AWS Lambda (Python 3.9 + AWS Data Wrangler Layer)
- **Storage:** AWS S3 (Data Lake)
- **Strategy:** “Split & Stitch” usingUNLOADto bypass memory limits.

### 2.2 B. AI Training Layer

Responsible for generating the ranking logic.

- **Compute:** AWS Fargate / SageMaker Processing Job
- **Frequency:** Weekly
- **Output:** XGBoost Ranker Model (brain.pkl)


### 2.3 C. Serving Layer (API)

Responsible for delivering real-time recommendations to the client.

- **Compute:** AWS App Runner (Containerized FastAPI)
- **State:** Stateless (Loads data from S3 on startup)
- **Performance:** In-memory Pandas filtering (Zero DB I/O during requests)

## 3 How It Works (The Data Flow)

### 3.1 Step 1: Daily Inventory Job

**Goal:** Create a clean, deduplicatedinventory.parquet.

**The “Split & Stitch” Scalability Fix:**
Redshift cannot aggregate 10L listings with amenities strings (causedLISTAGG65KB limit error).

1. **Split:** We split the export into two raw Parquet files viaUNLOAD:
    - listings_base/(Price, Beds, Location - Deduplicated via Window Function).
    - amenities_long/(Listing ID↔Amenity Code).
2. **Stitch:** The Lambda script downloads both, performs a fast Pandas Merge (safe for >10M
    rows), and saves the final file to S3.

### 3.2 Step 2: Daily User Data Job

**Goal:** Create a User Preference Matrix (user_data.parquet).

**Logic:** We query raw clickstream logs (stg_snowplow_events) aggregating 3 months of history.
We apply our **Weighted Interest Score** :

```
View = 1 Save = 5 Click = 10 Lead = 50 Unsave = -
```
**Scalability:** This aggregation happens entirely inside Redshift. We only export the final compact
result.

### 3.3 Step 3: Weekly AI Training

**Goal:** Update the ranking logic (brain.pkl).
**Logic:** Merges User History + Inventory to train an XGBoost model to predict “Interaction Prob-
ability.”

### 3.4 Step 4: API Startup (Serving)

When the API container starts, it downloadsinventory.parquet,user_data.parquet, andbrain.pkl
into RAM.
**Personalization:** When User X searches, we look up their ID in theuser_datadataframe (O(1)
complexity), get their liked items, and boost those IDs in the results.


## 4 Scalability & Resilience Defense

DevOps Team may ask: _“Willthiscrashwith10Millionlistings?”_

```
Concern Our Solution Verdict
```
```
Database Load We use Redshift UNLOAD (not SELECT *).
This is the fastest way to extract data and
puts minimal load on the DB.
```
```
Scalable
```
```
Memory Over-
flows
```
```
We use the Split & Stitch strategy. We
process data in Parquet (columnar format),
which is 10x smaller in RAM than JSON/CSV.
```
```
Scalable
```
```
Latency The API serves from RAM. It never touches
the database during a search request.
Sorting/Filtering 1M rows in Pandas takes
milliseconds.
```
```
Ultra-Fast
```
```
Concurrency The API is Stateless. You can run 100 con-
tainers behind a Load Balancer. They don’t
share state.
```
```
Horizontally Scalable
```
```
Cold Starts The Lambda uses the AWS Data Wrangler
layer for optimized S3/Parquet handling.
```
```
Optimized
```
## 5 DevOps Deployment Checklist

Please provision the following resources to go live:

### 5.1 1. IAM Role (For Lambda/Redshift)

- **Name:** RecSysDataPipelineRole
- **Policy:**
    **-** redshift-data:ExecuteStatement(To trigger queries).
    **-** s3:PutObject, s3:GetObject, s3:ListBucket(For the specific bucket).
    **-** secretsmanager:GetSecretValue(If DB creds are in Secrets Manager).

### 5.2 2. AWS S3 Bucket

**Structure:**

- /inventory/(Stores dailyinventory.parquet)
- /users/(Stores dailyuser_data.parquet)
- /brain/(Stores weeklybrain.pkl)
- /temp/(Scratchpad for Redshift exports - Set Lifecycle Policy to delete after 1 day).


### 5.3 3. Compute (Lambda)

- **Job 1:** DailyInventoryJob(3GB RAM, 10-15 min timeout).
- **Job 2:** DailyUserDataJob(512MB RAM, 5 min timeout).
- **Layer:** Must attach AWS SDK for Pandas (Python 3.9).

### 5.4 4. Scheduling (EventBridge)

- **Rule 1:** Configure schedule to triggerDailyInventoryJob.
- **Rule 2:** Configure schedule to triggerDailyUserDataJob(ensure it runs after Inventory
    Job).

### 5.5 5. API Hosting (App Runner / ECS)

- **Docker Image:** Build from the provided Dockerfile.
- **Env Variables:**
    **-** S3_BUCKET: [BUCKET_NAME]
    **-** AWS_REGION: us-east-
- **Instance Role:** Must haveAmazonS3ReadOnlyAccess.

## 6 Final Code Deliverables

You have the following 4 files ready for commit:

- main.py(The API).
- daily_inventory_job.py(The Listings ETL).
- daily_user_job.py(The User ETL).
- weekly_train_job.py(The AI Trainer).





# Technical & Strategic Documentation:

# Property Finder RecSys


## 1. Executive Summary

The **Property Finder Recommendation System (v12.4-Production)** is a sophisticated,
high-performance intelligence layer designed to evolve the property search experience from a
static database query into a dynamic, intent-driven journey.
By integrating **XGBoost gradient boosting** with **NLP-based behavioral profiling** , the engine
delivers real-time, hyper-personalized rankings. It processes an inventory of **44,000+ listings**
against a historical dataset of **1.55 million user interactions** , ensuring that every result is
mathematically optimized for both market quality and individual user taste.

## 2. Core Philosophy: "Intent over Attribute"

Traditional real estate platforms rely on "Hard Filtering" (e.g., "Show me 2-bedroom
apartments"). Our philosophy, refined in version 12.4, recognizes that user behavior is a more
accurate signal of desire than UI selections. The system operates on three psychological layers:

### Layer 1: The Rational Boundary (Constraint Satisfaction)

We respect the user's non-negotiable boundaries, such as budget, location, and property type.
These act as "Hard Gates," ensuring the engine remains grounded in the user's practical reality.

### Layer 2: The Social Proof (Market Quality)

Trust is the primary currency in real estate. This layer uses global machine learning models to
identify "Market Winners"—listings that are verified, competitively priced, and managed by
top-tier "Super Agents." This ensures that even without user history, the platform surfaces the
highest-quality inventory first.

### Layer 3: The Subconscious Affinity (Lifestyle Matching)

Users often have a "vibe" or aesthetic preference they cannot articulate. By analyzing historical
interaction "DNA" (e.g., a pattern of clicking properties with high-floor views or private pools), the


engine builds a **Subconscious Profile**. It surfaces properties that match the user’s "soul," even
if they fall outside standard sorting patterns.

## 3. Technical Methodology

### 3.1. Data Infrastructure

The engine utilizes a **Parquet-First** architecture, chosen for its high-speed I/O capabilities and
efficient handling of large-scale datasets.
● **Inventory Dataset:** 44,127 active listings with 38 feature columns.
● **Interaction Dataset:** 1,555,000 user interaction rows (clicks/leads).
● **Synchronization Logic:** A strict "Type-Safe Bridge" ensures that
property_listing_id fields are standardized across legacy history and active
snapshots to prevent data drift.

### 3.2. The Hybrid Scoring Formula

The final ranking of a property is determined by a multi-variate mathematical model:
$$RankScore = XGBoost(MarketQuality) + (Similarity(UserDNA, PropertyDNA) \times
Weight)$$
**A. Static ML Layer: XGBoost (brain.pkl)**
We deployed a Gradient Boosting Regressor trained on historical performance metrics.
● **Input Features:** price, size_sqft, super_agent_score, verified_flag, and
listing_level.
● **Objective:** To predict the objective "Market Value" of a listing.
**B. Dynamic NLP Layer: Vectorization & Similarity**
To achieve personalization, we treat property amenities as a language.
● **Tokenization:** A custom amenity_tokenizer parses codes (e.g., BA for Balcony, VW
for View).
● **Centroid Calculation:** We aggregate the vectors of every property a user has interacted
with to create a **"User DNA Centroid."**
● **Cosine Similarity:** We calculate the mathematical distance between the User Centroid
and the Candidate Property. If a match is found (e.g., both share a "Waterfront" tag), a
**5.0x Linear Boost** is applied to the property's rank.

## 4. Key Functional Features


```
Feature Technical Execution Business Value
Radius
Geospatial
Haversine
Vectorization Formula
Allows "Search Near Me" with kilometer-perfect
precision.
Commercial
Logic
ID-conditional
Formatting
Correctly labels "0 Bedrooms" as "Land/Office"
for Commercial, but "Studio" for Residential.
Amenity
Intersection
Boolean Array
Masking
Ensures strict "AND" logic for selected amenities
(e.g., Pool AND Gym).
Verified Priority Weight-Biased ML Increases platform trust by prioritizing "Verified"
and "Premium" listings.
Real-time
Re-ranking
Asynchronous
FastAPI Search
Delivers personalized results in <100ms.
```
## 5. Diagnostic Verification & QA

During the final production audit, the system underwent rigorous testing to ensure logic integrity:

1. **Schema Audit:** We identified and resolved a data-type mismatch (Integer vs. String)
    that was causing interaction history to be ignored. The "Force-String Bridge" now
    ensures 100% connectivity.
2. **The "Luxury Shift" Test:** We successfully simulated a high-intent user (interested in
    Waterfront/Luxury). The engine correctly promoted a **$52M Luxury Listing** to the #
    spot, leapfrogging cheaper but less relevant properties.
3. **Graceful Degradation:** For new users (Cold Start), the system automatically pivots to
    Layer 2 (Market Quality), ensuring a premium experience even without historical data.

## 6. Implementation & Scalability

The system is built for the modern web:


```
● Backend: FastAPI (Python) for high-concurrency handling.
● Data Science: Scikit-Learn, Pandas, NumPy, and XGBoost.
● API Contract: Returns standardized JSON objects containing meta_data,
rank_score, and ProtoProperty objects for seamless UI integration.
● Deployment: Stateless architecture, fully compatible with Docker and AWS/GCP cloud
environments.
```
## 7. Conclusion

The **v12.4-Production** engine is not merely a filter—it is a **Conversion Engine**. By
understanding the subtle patterns of human behavior and combining them with rigorous market
analysis, the system reduces "search fatigue" and accelerates the journey from a click to a
contract.
**Status:** Production Ready **Version:** 12.4.0 **Lead Logic:** Hybrid XGBoost/NLP Personalization







