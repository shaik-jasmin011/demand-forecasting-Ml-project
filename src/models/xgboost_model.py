"""
XGBoost Model for Demand Forecasting
Gradient-boosted trees using engineered tabular features — typically
the strongest performer on structured demand data.
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def train_xgboost(train_data, feature_cols, product_name="product"):
    """
    Train an XGBoost regressor on engineered features.
    
    Args:
        train_data: DataFrame with feature columns and 'demand'
        feature_cols: list of feature column names
        product_name: name for saving the model
    
    Returns:
        fitted XGBRegressor model
    """
    X_train = train_data[feature_cols].values
    y_train = train_data["demand"].values

    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"xgboost_{product_name}.pkl")
    joblib.dump(model, model_path)
    print(f"  📦 XGBoost model saved → {model_path}")

    return model


def predict_xgboost(model, data, feature_cols):
    """
    Generate demand predictions using trained XGBoost model.
    
    Args:
        model: fitted XGBRegressor
        data: DataFrame with feature columns
        feature_cols: list of feature column names
    
    Returns:
        numpy array of predictions
    """
    X = data[feature_cols].values
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)

    return predictions


def get_feature_importance(model, feature_cols):
    """
    Get feature importance from trained XGBoost model.
    
    Args:
        model: fitted XGBRegressor
        feature_cols: list of feature column names
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return df_importance
