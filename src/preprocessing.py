"""
Data Preprocessing & Feature Engineering Pipeline
Creates lag features, rolling statistics, temporal features, and train/test splits.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_data():
    """Load generated demand data."""
    df = pd.read_csv(config.GENERATED_DATA_FILE, parse_dates=["date"])
    df.sort_values(["product", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df):
    """Add engineered features for ML models."""
    df = df.copy()

    # Temporal features
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Holiday flag
    df["is_holiday"] = 0
    df.loc[(df["month"] == 12) & (df["day_of_month"].between(15, 31)), "is_holiday"] = 1
    df.loc[(df["month"] == 11) & (df["day_of_month"].between(24, 30)), "is_holiday"] = 1
    df.loc[(df["month"] == 1) & (df["day_of_month"] <= 3), "is_holiday"] = 1

    # Lag features (per product)
    for lag in [1, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby("product")["demand"].shift(lag)

    # Rolling statistics (per product)
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("product")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("product")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )

    # Expanding mean
    df["expanding_mean"] = (
        df.groupby("product")["demand"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Fill any NaN from lagging
    df.fillna(0, inplace=True)

    return df


def split_data(df, product=None):
    """
    Split data into train/test chronologically.
    Returns data for a specific product if specified.
    """
    if product:
        df = df[df["product"] == product].copy()

    cutoff_date = df["date"].max() - pd.Timedelta(days=config.TEST_SIZE_DAYS)
    train = df[df["date"] <= cutoff_date].copy()
    test = df[df["date"] > cutoff_date].copy()

    return train, test


def get_feature_columns():
    """Return list of feature columns for ML models."""
    features = [
        "price", "promotion", "day_of_week", "month", "is_weekend",
        "year", "quarter", "day_of_month", "week_of_year",
        "is_month_start", "is_month_end", "is_holiday",
    ]
    # Add lag features
    for lag in [1, 3, 7, 14, 30]:
        features.append(f"lag_{lag}")
    # Add rolling features
    for window in [7, 14, 30]:
        features.append(f"rolling_mean_{window}")
        features.append(f"rolling_std_{window}")
    features.append("expanding_mean")
    return features


def prepare_sequences(data, target_col="demand", seq_length=None):
    """Prepare sequences for LSTM model."""
    if seq_length is None:
        seq_length = config.LSTM_SEQUENCE_LENGTH

    scaler = MinMaxScaler()
    values = data[target_col].values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    return X, y, scaler


def preprocess_pipeline(product=None):
    """
    Full preprocessing pipeline.
    Returns train_df, test_df with all features added.
    """
    df = load_data()
    df = add_features(df)

    if product:
        train, test = split_data(df, product=product)
    else:
        train, test = split_data(df)

    print(f"✅ Preprocessing complete | Train: {len(train):,} | Test: {len(test):,}")
    return train, test


if __name__ == "__main__":
    train, test = preprocess_pipeline("Electronics")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"\nTrain sample:\n{train.head()}")
