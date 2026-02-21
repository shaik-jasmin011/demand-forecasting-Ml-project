# 📊 DemandAI — Demand Forecasting & Optimization Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/shaik-jasmine/demand-forecasting-ml-project/main/dashboard/app.py)

A production-ready demand forecasting system that uses **5 ML models** on real e-commerce data to predict future demand, optimize inventory, and plan logistics — all visualized through an interactive Streamlit dashboard.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **5 ML Models** | ARIMA, Prophet, XGBoost, LSTM, and Ensemble (XGBoost + Prophet) |
| **Real Data** | UCI Online Retail II dataset (~1M transactions) with automatic download |
| **Interactive Dashboard** | 5-page Streamlit app with dark theme and Plotly charts |
| **Inventory Optimization** | EOQ, safety stock, reorder points, and ABC classification |
| **Logistics Planning** | Shipment scheduling, warehouse allocation, route priority scoring |
| **Feature Importance** | XGBoost-based analysis with cross-product heatmap |
| **Hyperparameter Tuning** | Optuna-powered optimization for XGBoost and LSTM |
| **Report Export** | One-click HTML report generation from the dashboard |
| **Docker Support** | Containerized deployment with Docker Compose |

---

## 🏗️ Architecture

```
demand-forecasting-project/
│
├── config.py                 # Centralized configuration
├── train.py                  # Main training pipeline
│
├── src/
│   ├── data_generator.py     # Synthetic data generator
│   ├── real_data_loader.py   # UCI Online Retail II loader
│   ├── preprocessing.py      # Feature engineering pipeline
│   ├── evaluation.py         # Model metrics & comparison
│   ├── inventory_optimizer.py# EOQ, safety stock, ABC
│   ├── logistics_optimizer.py# Shipments, warehouses, routing
│   ├── report_generator.py   # HTML report builder
│   ├── tuner.py              # Optuna hyperparameter tuning
│   └── models/
│       ├── arima_model.py    # SARIMA time series
│       ├── prophet_model.py  # Facebook Prophet
│       ├── xgboost_model.py  # Gradient boosted trees
│       ├── lstm_model.py     # Deep learning (TensorFlow)
│       └── ensemble_model.py # Weighted XGBoost + Prophet
│
├── dashboard/
│   └── app.py                # Streamlit dashboard (5 pages)
│
├── tests/
│   └── test_pipeline.py      # Pytest test suite
│
├── data/                     # Generated/downloaded data
├── models_saved/             # Trained model artifacts
├── results/                  # Predictions, reports, metrics
│
├── Dockerfile                # Container image
├── docker-compose.yml        # Multi-service deployment
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/demand-forecasting-project.git
cd demand-forecasting-project
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train.py
```

This will:
- Download the UCI Online Retail II dataset (~44 MB)
- Clean and categorize ~1M transactions into 5 product groups
- Train 5 models (ARIMA, Prophet, XGBoost, LSTM, Ensemble) per category
- Evaluate with MAE, RMSE, MAPE, R²
- Run inventory and logistics optimization
- Save all results to `results/`

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## 📈 Dashboard Pages

| Page | What It Shows |
|------|--------------|
| 📈 **Demand Forecast** | Historical trends, seasonal patterns, model predictions |
| 📦 **Inventory Optimizer** | EOQ, safety stock, cost breakdown, ABC classification |
| 🚚 **Logistics Planner** | Shipment schedule, warehouse allocation, route priority |
| 📊 **Model Comparison** | RMSE/R² bars, radar chart, residual analysis |
| 🔍 **Feature Importance** | All features ranked, top-10 breakdown, cross-product heatmap |
| 📄 **Report Download** | One-click HTML report (sidebar button) |

---

## 🔧 Configuration

All settings are in `config.py`:

```python
USE_REAL_DATA = True          # True = UCI data | False = synthetic
FORECAST_HORIZON = 30         # Days to forecast
TEST_SIZE_DAYS = 90           # Test set size
SARIMA_ORDER = (1, 1, 1)      # ARIMA parameters
XGB_PARAMS = {                # XGBoost parameters
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    ...
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Tests cover:
- Data loading and validation
- Preprocessing and feature engineering
- XGBoost training and prediction
- Ensemble weight optimization
- Config integrity
- Report generation

---

## ⚡ Hyperparameter Tuning

Optimize model parameters with Optuna:

```bash
# Tune XGBoost for all products (50 trials each)
python -m src.tuner --model xgboost --trials 50

# Tune LSTM (20 trials)
python -m src.tuner --model lstm --trials 20

# Tune specific product
python -m src.tuner --product "Home & Living" --trials 100
```

Best parameters are saved to `results/` and can be loaded in future training runs.

---

## 🐳 Docker

### Build & Run Dashboard

```bash
docker-compose up -d
```

### Run Training Pipeline

```bash
docker-compose --profile train run train
```

Access dashboard at **http://localhost:8501**

---

## 📊 Model Performance (Real Data)

| Product | Best Model | MAE | RMSE | R² |
|---------|-----------|-----|------|-----|
| Home & Living | XGBoost | 1,523 | 1,992 | 0.76 |
| Kitchen & Dining | XGBoost | 1,194 | 1,655 | 0.44 |
| Gifts & Party | XGBoost | 1,422 | 2,002 | 0.67 |
| Garden & Outdoor | Prophet | 1,196 | 6,951 | 0.33 |
| Accessories & Fashion | XGBoost | 466 | 621 | 0.56 |

> **Note:** The Ensemble model (XGBoost + Prophet) typically improves upon individual models by 3-8% on RMSE.

---

## 🔑 Key Features Deep Dive

### Feature Importance (Home & Living)
| Feature | Importance |
|---------|-----------|
| promotion | 51.2% |
| is_weekend | 17.2% |
| is_holiday | 15.2% |
| day_of_week | 3.6% |
| rolling_mean_7 | 1.9% |

### Product Categories
The real data loader categorizes ~4,000 product descriptions into 5 groups using keyword matching:
- **Home & Living** — candles, cushions, frames, clocks, lamps
- **Kitchen & Dining** — mugs, plates, bowls, tea, coffee
- **Gifts & Party** — gift cards, Christmas, birthdays, wrapping
- **Garden & Outdoor** — plants, flowers, pots, birds, solar
- **Accessories & Fashion** — bags, jewelry, scarves, watches

---

## 📁 Data

- **Real Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — ~1M transactions from a UK online retailer (2009–2011)
- **Synthetic Data:** Built-in generator with configurable parameters

Toggle in `config.py`:
```python
USE_REAL_DATA = True   # Real data
USE_REAL_DATA = False  # Synthetic data
```

---

## 🛠️ Tech Stack

- **ML/DL:** scikit-learn, XGBoost, Prophet, TensorFlow/Keras, statsmodels
- **Tuning:** Optuna
- **Visualization:** Plotly, Streamlit
- **Data:** pandas, NumPy
- **Deployment:** Docker, Docker Compose
- **Testing:** pytest

---

## 📄 License

MIT License — free for personal and commercial use.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  <b>Built with ❤️ using Python, Streamlit & Machine Learning</b>
</p>
