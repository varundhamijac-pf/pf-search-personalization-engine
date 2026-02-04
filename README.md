# Property Finder Recommendation Engine (v6.6.0)

> **Status:** Production Ready (Phase 1)  
> **Type:** Hybrid Filtering + Learning-to-Rank (LTR) AI  
> **Tech Stack:** Python, FastAPI, XGBoost, Pandas, Parquet

---

## 1. Executive Summary

This project represents a shift from traditional "Database Search" (SQL WHERE clauses) to an **AI-Driven Discovery Engine**.

Instead of simply returning listings that match a filter, this engine calculates the **Probability of Conversion** for every listing. It asks: *"Among the 500 apartments matching the user's criteria, which ones are most likely to result in a Lead?"*

It achieves this through a **3-Layer Architecture**:
1.  **Strict Filtering:** Hard constraints (Price, Beds, Location).
2.  **Geospatial Intelligence:** Radius-based search with text fallbacks.
3.  **AI Re-Ranking:** XGBoost model predicting user interest based on behavioral signals.

---

## 2. System Architecture

### A. The Data Pipeline (`pipeline.py`)
We moved away from CSVs to **Parquet** for performance. The pipeline handles:
* **Ingestion:** Merging raw listing data with 90-day behavioral signals (Views, Clicks, Leads).
* **Normalization:** Cleaning dirty data (e.g., mapping `furnished: "Yes"` $\to$ `1`, `completion: "Ready"` $\to$ `completed`).

### B. The AI Brain (`light_brain.pkl`)
* **Model:** XGBoost Classifier (Optimized for Speed).
* **Objective:** Binary Classification (Target: `lead_submission_count > 0`).
* **Key Features:**
    * `smart_popularity_score` (Weighted interaction metric).
    * `price` (Relative to market).
    * `freshness_score` (Time decay).
    * `trust_score` (Agent quality).

### C. The API (`app_7.py`)
A high-performance **FastAPI** service that loads the Data and Brain into **RAM** for sub-50ms inference.

---

## 3. The "Strict but Smart" Logic

A major challenge in Real Estate search is handling messy data (e.g., missing coordinates) without returning empty results. We implemented a robust fallback chain.

### Layer 1: Hard Filters (The "Strict" Part)
We apply strict boolean masks for critical user requirements. If a user asks for "2 Beds", we never show 1 Bed.
* **Filters:** `category_id` (Buy/Rent), `price_range`, `bedrooms`, `bathrooms`, `furnished_status`.

### Layer 2: Geospatial Intelligence (The "Smart" Part)
Users search by **Keyword** (e.g., "Marina"), but computers need **Coordinates**.
1.  **Centroid Lookup:** The engine checks if "Marina" matches a known location centroid.
2.  **Radius Search:** If matched, it calculates Haversine Distance and fetches listings within **3.5km**.

**The Fallback Safety Net:**
* **Scenario:** A listing is in Marina but has `lat: 0.0, lon: 0.0` (bad data).
* **Fix:** The Radius search fails (0 results). The engine detects this and automatically switches to **Text Match** (`WHERE location_name LIKE '%Marina%'`), ensuring the user still gets results.

### Layer 3: AI Re-Ranking (Learning-to-Rank)
Once Layer 1 & 2 narrow the pool (e.g., from 72,000 $\to$ 200 listings), the AI takes over.
* **Feature Extraction:** The engine builds a feature vector for the 200 candidates.
* **Inference:** The XGBoost model predicts a score ($0.0 \to 1.0$) representing "Lead Probability".

**Scoring Logic:**
$$
\text{Final Score} = (\text{AI Probability} \times 100) - (\text{Distance} \times 0.1) + \text{Luxury Boost}
$$

**Result:** A popular, high-quality listing 1km away ranks higher than a stale listing 0.5km away.

---

## 4. Deployment Status (Phase 1)

| Feature | Status | Notes |
| :--- | :---: | :--- |
| **Basic Search** | Live | Filters by Price, Beds, Area working perfectly. |
| **Geo Search** | Live | Includes Radius Search + "Zero Coordinate" Fallback. |
| **AI Ranking** | Live | "Featured" sort uses XGBoost probability. |
| **Price Period** | Hidden | Logic exists in DB but API filter is disabled for stability. |
| **Performance** | High | In-Memory Parquet loads < 1s, Query < 50ms. |

---

## 5. Phase 2 Roadmap: Moving to "Industry Grade"

While Phase 1 is a robust Search Engine, Phase 2 transforms it into a **Personalization Engine**.

### Objective 1: User-Level Personalization
Currently, every user sees the same "Popular" results.
* **The Plan:** Implement a "Shadow Profile" using Redis.
* **Logic:**
    * If User A clicks 3 Villas $\to$ Boost Villa scores by 20%.
    * If User B filters "Price > 5M" $\to$ Tag as "Luxury User" and deprioritize cheap listings.

### Objective 2: Hierarchical Location Search
Currently, "Dubai" search relies on radius or text match.
* **The Plan:** Ingest `full_location_path` (e.g., Dubai - Marina - Princess Tower).
* **Logic:** Use Path Containment (`path LIKE 'Dubai%'`) to officially capture all child communities without relying on distance.

### Objective 3: Rental Period Logic (`price_type`)
* **The Plan:** Expose the `price_period` filter in the API.
* **Logic:**
    * Allow users to toggle "Daily" vs "Yearly".
    * Prevent 500 AED (Daily) listings from cluttering a "Cheap Yearly Rent" search.

---

## 6. API Interface

**Endpoint:** `POST /api/v1/search`

The API accepts a strict JSON structure.

### Sample Payload
```json
{
  "filters": {
    "category_id": 2,
    "min_price": 50000,
    "max_price": 150000,
    "keywords": ["Marina"],
    "number_of_bedrooms": [1, 2],
    "is_super_agent": false
  },
  "pagination": {
    "page": 1,
    "limit": 20
  },
  "sorting": {
    "sort": "featured"
  }
}
