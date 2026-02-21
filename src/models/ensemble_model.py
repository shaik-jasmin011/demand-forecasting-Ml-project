"""
Ensemble Model — Weighted Average of XGBoost + Prophet
Combines the strengths of both models for more robust demand forecasting.
The weights are optimized to minimize RMSE on the training residuals.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def find_optimal_weights(y_true, pred_xgb, pred_prophet, steps=101):
    """
    Grid search for the best weight α such that:
        ensemble = α * XGBoost + (1-α) * Prophet
    minimizes RMSE on the given actuals.
    """
    best_rmse = np.inf
    best_alpha = 0.5

    for i in range(steps):
        alpha = i / (steps - 1)
        blended = alpha * pred_xgb + (1 - alpha) * pred_prophet
        rmse = np.sqrt(np.mean((y_true - blended) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    return best_alpha, best_rmse


def train_ensemble(y_true, pred_xgb, pred_prophet, product_name="product"):
    """
    Find optimal ensemble weights and save them.

    Args:
        y_true: actual demand values (numpy array)
        pred_xgb: XGBoost predictions (numpy array)
        pred_prophet: Prophet predictions (numpy array)
        product_name: name for saving weights

    Returns:
        dict with alpha, rmse, and blended predictions
    """
    alpha, rmse = find_optimal_weights(y_true, pred_xgb, pred_prophet)
    blended = alpha * pred_xgb + (1 - alpha) * pred_prophet
    blended = np.maximum(blended, 0)

    # Save weights
    weights = {"alpha": alpha, "rmse": rmse, "product": product_name}
    weight_path = os.path.join(config.MODEL_DIR, f"ensemble_weights_{product_name}.pkl")
    joblib.dump(weights, weight_path)
    print(f"  📦 Ensemble weights saved → {weight_path}")
    print(f"     α={alpha:.2f} (XGBoost={alpha*100:.0f}%, Prophet={100-alpha*100:.0f}%), RMSE={rmse:.2f}")

    return {"alpha": alpha, "rmse": rmse, "predictions": blended}


def predict_ensemble(pred_xgb, pred_prophet, product_name="product"):
    """
    Apply saved ensemble weights to new predictions.
    Falls back to equal weighting if no saved weights found.
    """
    weight_path = os.path.join(config.MODEL_DIR, f"ensemble_weights_{product_name}.pkl")

    if os.path.exists(weight_path):
        weights = joblib.load(weight_path)
        alpha = weights["alpha"]
    else:
        alpha = 0.5  # fallback

    blended = alpha * pred_xgb + (1 - alpha) * pred_prophet
    return np.maximum(blended, 0)
