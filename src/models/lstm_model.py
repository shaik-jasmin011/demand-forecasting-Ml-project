"""
LSTM Neural Network for Demand Forecasting
Sequence-based deep learning approach using sliding windows.
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

warnings.filterwarnings("ignore")

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train_lstm(train_data, product_name="product"):
    """
    Train an LSTM model on demand time series using sliding windows.
    
    Args:
        train_data: DataFrame with 'demand' column (sorted by date)
        product_name: name for saving the model
    
    Returns:
        trained Keras model, fitted scaler
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    # Scale data
    scaler = MinMaxScaler()
    demand_values = train_data["demand"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(demand_values)

    # Create sequences
    seq_len = config.LSTM_SEQUENCE_LENGTH
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    # Build model
    model = Sequential([
        LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(config.LSTM_UNITS // 2, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True,
    )

    model.fit(
        X, y,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    # Save model and scaler
    model_path = os.path.join(config.MODEL_DIR, f"lstm_{product_name}.keras")
    scaler_path = os.path.join(config.MODEL_DIR, f"lstm_scaler_{product_name}.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  📦 LSTM model saved → {model_path}")

    return model, scaler


def predict_lstm(model, scaler, data, seq_length=None):
    """
    Generate predictions using trained LSTM model.
    
    Args:
        model: trained Keras LSTM model
        scaler: fitted MinMaxScaler
        data: DataFrame with 'demand' column (includes history for sequences)
        seq_length: length of input sequences
    
    Returns:
        numpy array of predictions (last len(data)-seq_length values)
    """
    if seq_length is None:
        seq_length = config.LSTM_SEQUENCE_LENGTH

    demand_values = data["demand"].values.reshape(-1, 1)
    scaled = scaler.transform(demand_values)

    X = []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length:i, 0])

    X = np.array(X).reshape(-1, seq_length, 1)

    scaled_preds = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(scaled_preds).flatten()
    predictions = np.maximum(predictions, 0)

    return predictions


def predict_lstm_future(model, scaler, last_sequence, steps=None):
    """
    Generate future rolling forecasts.
    
    Args:
        model: trained Keras LSTM model
        scaler: fitted MinMaxScaler
        last_sequence: numpy array of the last `seq_length` demand values
        steps: number of future steps to predict
    
    Returns:
        numpy array of future predictions
    """
    if steps is None:
        steps = config.FORECAST_HORIZON

    seq_length = len(last_sequence)
    scaled_seq = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    current = list(scaled_seq)

    predictions = []
    for _ in range(steps):
        x_input = np.array(current[-seq_length:]).reshape(1, seq_length, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        current.append(pred_scaled)
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(max(pred, 0))

    return np.array(predictions)
