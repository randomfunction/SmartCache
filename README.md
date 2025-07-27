
# Intelligent ML-Driven Cache System (Offline + Online)

A high-performance hybrid caching system that combines traditional policies (LRU, LFU, MFU) with both **offline-trained** and **online-learning** machine learning models to approximate **Belady's Optimal Replacement Policy**.

---

## Overview

This project explores intelligent caching by:

* Generating realistic **Zipfian (skewed)** access patterns
* Simulating **Belady’s optimal replacement** using full and partial lookahead
* Training ML models (Logistic Regression, XGBoost, CatBoost, LightGBM) to **predict cache-worthiness**
* Implementing a real-time **Online ML Cache** using the `river` library
* Benchmarking all strategies against LRU, LFU, MFU

---

## Benchmark Results (Offline ML Cache)

| Cache Type                  | Hit Rate   |
| --------------------------- | ---------- |
| LRU                         | 0.3402     |
| LFU                         | 0.4776     |
| MFU                         | 0.0960     |
| **ML (Best Offline Model)** | **0.4805** |

---

## Online Learning Results

| Metric                  | Value  |
| ----------------------- | ------ |
| **Streaming Hit Rate**  | 0.6611 |
| **Prediction Accuracy** | 0.9509 |

> The online ML cache adapts to access patterns in real time, significantly outperforming all static policies.

---

## Features

* Custom implementation of **LRU**, **LFU**, and **MFU**
* Training and selection of best model among **LogReg**, **XGBoost**, **CatBoost**, **LightGBM**
* Belady’s **optimal replacement simulation** (full and partial lookahead)
* Real-time **OnlineMLCache** (built with `river`) that trains on-the-fly
* Hit rate + accuracy tracking and full benchmarking


## Setup & Usage


### 1. Install all dependencies
pip install -r requirements.txt

### 2. Generate labeled data (offline Belady)
python data_generator.py --num_requests 100000 --unique_items 1000 --capacity 100 --output data/labeled_requests.csv

### 3. Train all ML models
python train_models.py

### 4. Run offline ML cache
python test_cache.py

### 5. Benchmark all strategies (LRU, LFU, MFU, ML)
python compare_cache.py

### 6. Run online ML cache (real-time learning)
python test_online_cache.py


---

## Sample Output

```bash
=== Offline Cache Hit Rates ===
LRU : 0.3402
LFU : 0.4776
MFU : 0.0960
ML  : 0.4805

--- Streaming Cache ---
Streaming ML-Cache Hit Rate: 0.6611
Prediction Accuracy:       0.9509
```

## License

MIT License.

---

## 👤 Author

**Tanishq Parihar**
*Built as a systems + ML hybrid project to explore intelligent infrastructure and predictive caching*

---
