# 📈 XAlgo – Triangular Arbitrage Intelligence Engine

**XAlgo** is a quant-grade, real-time arbitrage engine built for triangular trading opportunities in both **crypto** and **forex** markets.
It combines adaptive statistics, machine learning, and signal execution logic to capture inefficiencies across interconnected trading pairs like BTC/ETH/USDT and GBP/USD/EUR.

---

## 🚀 System Overview

### ✅ Live Signal Engine Highlights

* **Real-time Binance stream ingestion**
* **Feature generation**: Z-score, EWMA, Kalman Filter, spread slope
* **Statistical gates**: Z-score + volatility filtering
* **ML Filters**:

  * Directional confidence filter (multi-class)
  * Cointegration stability classifier
  * Optimal trade leg selector (BTC or ETH)
  * Market regime classifier (flat, volatile, trending)
* **Dynamic SL/TP** based on regime, spread, and volatility
* **Cluster guard** to avoid overlapping trades

---

## 🤖 Models Used

| Model File                      | Role                                                    |
| ------------------------------- | ------------------------------------------------------- |
| `triangular_rf_model.pkl`       | Directional classifier (–1 = SHORT, 0 = HOLD, 1 = LONG) |
| `cointegration_score_model.pkl` | Classifies spread stability via Kalman RMS              |
| `pair_selector_model.pkl`       | Predicts best trading leg: BTC, ETH, or neutral         |
| `regime_classifier.pkl`         | Classifies volatility regimes                           |

---

## 📂 Project Structure

```bash
.
├── core/
│   ├── feature_pipeline.py           # Live + batch feature generator
│   ├── execution_engine.py           # Handles trade logic, SL/TP
│   ├── adaptive_filters.py           # EWMA + Kalman smoothing
│   ├── kalman_cointegration_monitor.py
├── utils/
│   └── filters.py                    # MLFilter wrapper
├── data/
│   ├── BTCUSDT.csv
│   ├── ETHUSDT.csv
│   ├── ETHBTC.csv
├── ml_model/
│   ├── triangular_rf_model.pkl/json
│   ├── cointegration_score_model.pkl/json
│   ├── pair_selector_model.pkl/json
│   ├── regime_classifier.pkl/json
├── logs/
│   └── signal_log.csv
├── scripts/
│   ├── generate_features_triangular_base.py
│   ├── label_all_targets.py
│   └── retrain_all_models.py
├── main.py
└── README.md
```

---

## 🥪 Training Pipeline

Train all 4 models using:

```bash
python retrain_all_models.py
```

### 📁 Uses:

* `features_triangular_base.csv` → base features
* `label_all_targets.py` → generates:

  * `features_triangular_labeled.csv`
  * `features_cointegration_labeled.csv`
  * `features_pair_selection.csv`
  * `features_regime.csv`

---

## ⚙️ Signal Flow Diagram

```text
[Live Binance Prices]
        ↓
[Feature Generator (live or batch)]
        ↓
[Z-Score + Slope Filter] → [Regime Classifier]
        ↓
[ML Confidence Model] → veto HOLDs
        ↓
[Cointegration Classifier] → veto unstable spreads
        ↓
[Best-Leg Selector] → BTC or ETH
        ↓
[Trade Execution (with SL/TP)] → Cluster Guard → Trade Logger
```

---

## 📊 Signal Logs

* Saved to: `logs/signal_log.csv`
* Columns:

  * `timestamp, spread, confidence, zscore, direction, action, reason, profit, sl, tp, slope, regime`
* Set environment variable `DISPLAY_HOLD=false` to suppress HOLD log lines.

---

## 📡 Monitoring & Automation (In Progress)

* ✅ Telegram & Discord alerts
* ✅ Prometheus metrics: confidence, cointegration, regime
* ✅ Grafana dashboards
* ✅ Live capital execution via Binance API

---

## 🧐 Engineered By

> Owen Mbugua · Quant Engineer @ XAlgo · 2025
> Passionate about market microstructure, machine learning, and real-time execution intelligence.

---

## 📥 Quick Start

### Run live signal engine:

```bash
python main.py
```

### Run full model training:

```bash
python retrain_all_models.py
```

### Or label all training data from base:

```bash
python label_all_targets.py
```

### Parameter sweep backtest

Use the provided `run_backtest.py` script to evaluate different confidence and
SL/TP thresholds. Threshold ranges are loaded from `config/default.yaml` so runs
are fully reproducible.

```bash
python run_backtest.py
```

---

## 💡 Notes

* Ensure you have raw CSVs in `data/` for BTCUSDT, ETHUSDT, ETHBTC
* Model `.pkl` files will be auto-generated in `ml_model/`
* System is modular and ready for FX (EURUSD, GBPUSD, EURGBP) with minor changes

---
