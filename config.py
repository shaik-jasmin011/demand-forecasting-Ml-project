"""
Central Configuration for Demand Forecasting Platform
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models_saved")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Data Mode ────────────────────────────────────────────────────────────────
USE_REAL_DATA = True  # True = UCI Online Retail II | False = synthetic data

REAL_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"

# ─── Data Generation ─────────────────────────────────────────────────────────
SYNTHETIC_PRODUCTS = [
    "Electronics",
    "Clothing",
    "Groceries",
    "Furniture",
    "Sports Equipment",
]

REAL_PRODUCTS = [
    "Home & Living",
    "Kitchen & Dining",
    "Gifts & Party",
    "Garden & Outdoor",
    "Accessories & Fashion",
]

# Active product list based on mode
PRODUCTS = REAL_PRODUCTS if USE_REAL_DATA else SYNTHETIC_PRODUCTS

DATA_START_DATE = "2022-01-01"
DATA_END_DATE = "2024-12-31"
GENERATED_DATA_FILE = os.path.join(DATA_DIR, "demand_data.csv")

# ─── Model Training ──────────────────────────────────────────────────────────
FORECAST_HORIZON = 30          # days to forecast into the future
TEST_SIZE_DAYS = 90             # last N days used as test set
RANDOM_STATE = 42

# ARIMA / SARIMA
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)   # weekly seasonality

# XGBoost
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# LSTM
LSTM_SEQUENCE_LENGTH = 30
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 64

# ─── Inventory Optimization ──────────────────────────────────────────────────
HOLDING_COST_PER_UNIT = 2.50          # $ per unit per year
ORDERING_COST_PER_ORDER = 50.00       # $ per order
LEAD_TIME_DAYS = 7                    # days from order to delivery
SERVICE_LEVEL = 0.95                  # 95% service level for safety stock

# ─── Logistics Optimization ──────────────────────────────────────────────────
WAREHOUSES = ["Warehouse A", "Warehouse B", "Warehouse C"]
SHIPPING_COST_PER_UNIT_PER_KM = 0.05  # $ per unit per km
WAREHOUSE_CAPACITIES = {
    "Warehouse A": 10000,
    "Warehouse B": 8000,
    "Warehouse C": 12000,
}
WAREHOUSE_DISTANCES = {
    # distance to demand center in km
    "Warehouse A": 150,
    "Warehouse B": 280,
    "Warehouse C": 90,
}
