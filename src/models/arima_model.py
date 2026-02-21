"""
ARIMA / SARIMA Model for Demand Forecasting
Captures autoregressive patterns and seasonal demand cycles.
"""
import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

warnings.filterwarnings("ignore")


def train_arima(train_data, product_name="product"):
    """
    Train a SARIMA model on demand time series.
    
    Args:
        train_data: DataFrame with 'date' and 'demand' columns
        product_name: name for saving the model
    
    Returns:
        fitted SARIMAX model
    """
    series = train_data.set_index("date")["demand"].asfreq("D")
    series = series.fillna(method="ffill").fillna(0)

    model = SARIMAX(
        series,
        order=config.SARIMA_ORDER,
        seasonal_order=config.SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False, maxiter=200)

    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"arima_{product_name}.pkl")
    joblib.dump(fitted, model_path)
    print(f"  📦 ARIMA model saved → {model_path}")

    return fitted


def predict_arima(model, steps=None):
    """
    Generate demand forecast using trained ARIMA model.
    
    Args:
        model: fitted SARIMAX model
        steps: number of days to forecast
    
    Returns:
        numpy array of predictions
    """
    if steps is None:
        steps = config.TEST_SIZE_DAYS

    forecast = model.forecast(steps=steps)
    predictions = np.maximum(forecast.values, 0)  # no negative demand

    return predictions


def predict_arima_insample(model, test_data):
    """
    Generate in-sample predictions aligned with test data dates.
    
    Args:
        model: fitted SARIMAX model
        test_data: DataFrame with 'date' column
    
    Returns:
        numpy array of predictions
    """
    start = test_data["date"].iloc[0]
    end = test_data["date"].iloc[-1]

    predictions = model.predict(start=start, end=end)
    predictions = np.maximum(predictions.values, 0)

    return predictions[:len(test_data)]
