"""
Facebook Prophet Model for Demand Forecasting
Handles holidays, custom seasonality, and promotional regressors.
"""
import os
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

warnings.filterwarnings("ignore")


def train_prophet(train_data, product_name="product"):
    """
    Train a Prophet model on demand data.
    
    Args:
        train_data: DataFrame with 'date', 'demand', and optionally 'promotion' columns
        product_name: name for saving the model
    
    Returns:
        fitted Prophet model
    """
    # Prophet requires 'ds' and 'y' columns
    df_prophet = train_data[["date", "demand"]].copy()
    df_prophet.columns = ["ds", "y"]

    # Add promotion as a regressor if available
    if "promotion" in train_data.columns:
        df_prophet["promotion"] = train_data["promotion"].values

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95,
    )

    # Add custom seasonality
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # Add promotion regressor
    if "promotion" in df_prophet.columns:
        model.add_regressor("promotion")

    model.fit(df_prophet)

    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"prophet_{product_name}.pkl")
    joblib.dump(model, model_path)
    print(f"  📦 Prophet model saved → {model_path}")

    return model


def predict_prophet(model, test_data=None, periods=None):
    """
    Generate demand forecast using trained Prophet model.
    
    Args:
        model: fitted Prophet model
        test_data: DataFrame with 'date' (and optionally 'promotion') for in-sample prediction
        periods: number of future periods to forecast (used if test_data is None)
    
    Returns:
        numpy array of predictions, DataFrame with full forecast details
    """
    if test_data is not None:
        future = pd.DataFrame({"ds": test_data["date"].values})
        if "promotion" in test_data.columns:
            future["promotion"] = test_data["promotion"].values
        else:
            future["promotion"] = 0
    else:
        if periods is None:
            periods = config.FORECAST_HORIZON
        future = model.make_future_dataframe(periods=periods)
        future["promotion"] = 0
        future = future.tail(periods).reset_index(drop=True)

    forecast = model.predict(future)
    predictions = np.maximum(forecast["yhat"].values, 0)

    return predictions, forecast
