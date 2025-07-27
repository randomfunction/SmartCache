# Intelligent ML-Driven Cache System

A high-performance hybrid caching system that uses traditional policies (LRU, LFU, MFU) and compares them with a **Machine Learning-based intelligent caching strategy** trained to approximate **Belady's Optimal Replacement Policy**.

---

## Overview

This project explores intelligent caching by:

- Generating realistic skewed access patterns using Zipf distribution
- Labeling optimal cache behavior using Belady's algorithm
- Training multiple ML models to predict cache-worthy items
- Integrating the best-performing model into a custom cache system
- Benchmarking against traditional strategies (LRU, LFU, MFU)

---

## Benchmark Results

| Cache Type | Hit Rate |
|------------|----------|
| LRU        | 0.3402   |
| LFU        | 0.4776   |
| MFU        | 0.0960   |
| **ML (Best Model)** | **0.4805** |

> ML-based cache slightly outperforms LFU, showcasing learning-based adaptation from access history.

---

## Features

- Custom implementation of **LRU**, **LFU**, and **MFU** cache policies
- Trains 4 ML models (LogReg, XGBoost, CatBoost, LightGBM)
- Selects the **best model automatically based on validation accuracy**
- Simulates **Belady’s optimal policy** for labeling training data
- Logs and prints hit rate comparisons across policies

---

## File Structure

```
.
├── data_generator.py       # Generates skewed access patterns + labels with Belady
├── train_models.py         # Extracts features and trains models
├── test_cache.py           # Tests smart cache with best model on fresh data
├── compare_cache.py        # Benchmarks LRU, LFU, MFU, ML on the same data
├── models/                 # Stores trained ML models
├── data/                   # Contains generated labeled data (CSV)
└── requirements.txt        # Python dependencies
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate labeled training data
python data_generator.py --num_requests 100000 --unique_items 1000 --capacity 100 --output data/labeled_requests.csv

# Train models and save the best
python train_models.py

# Test smart cache performance
python test_cache.py

# Benchmark all cache strategies
python compare_cache.py
```

---

## Sample Output

```
=== Cache Hit Rates ===
LRU : 0.3402
LFU : 0.4776
MFU : 0.0960
ML  : 0.4805
```

---

## TODOs & Future Work

- Add recency-based features (e.g., time since last seen)
- Use online learning to update model during simulation
- Visualize model decisions and cache evictions
- Dockerize the project for reproducibility

---

## License
MIT License.

---

## Author
Tanishq Parihar — *Built as a systems+ML project to explore learning-enhanced infrastructure*
