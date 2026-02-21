"""
Hyperparameter Tuner — Optuna-based optimization for XGBoost and LSTM.
Run standalone or integrate into the training pipeline for automatic tuning.

Usage:
    python -m src.tuner                   # tune XGBoost for all products
    python -m src.tuner --model lstm      # tune LSTM
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _load_product_data(product):
    """Load and prepare data for a single product."""
    from src.preprocessing import load_data, add_features, split_data, get_feature_columns

    df = load_data()
    df = add_features(df)
    train, test = split_data(df, product=product)
    feature_cols = get_feature_columns(train)
    return train, test, feature_cols


# ─── XGBoost Tuning ──────────────────────────────────────────────────────────

def tune_xgboost(product, n_trials=50):
    """Optimize XGBoost hyperparameters with Optuna."""
    from xgboost import XGBRegressor

    train, test, feature_cols = _load_product_data(product)
    X_train = train[feature_cols].values
    y_train = train["demand"].values
    X_test = test[feature_cols].values
    y_test = test["demand"].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        return rmse

    study = optuna.create_study(direction="minimize", study_name=f"xgb_{product}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["random_state"] = 42
    print(f"\n🏆 Best XGBoost params for {product}:")
    print(f"   RMSE: {study.best_value:.2f}")
    for k, v in best.items():
        print(f"   {k}: {v}")

    # Save best params
    import joblib
    param_path = os.path.join(config.RESULTS_DIR, f"best_xgb_params_{product}.pkl")
    joblib.dump(best, param_path)
    print(f"   Saved → {param_path}")

    return best, study.best_value


# ─── LSTM Tuning ──────────────────────────────────────────────────────────────

def tune_lstm(product, n_trials=20):
    """Optimize LSTM hyperparameters with Optuna."""
    from sklearn.preprocessing import MinMaxScaler
    import warnings
    warnings.filterwarnings("ignore")

    # Suppress TF logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    train, test, _ = _load_product_data(product)

    train_demand = train["demand"].values.reshape(-1, 1)
    test_demand = test["demand"].values.reshape(-1, 1)
    all_demand = np.concatenate([train_demand, test_demand])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(all_demand).flatten()

    def objective(trial):
        seq_len = trial.suggest_int("sequence_length", 7, 60, step=7)
        units = trial.suggest_int("units", 32, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        epochs = trial.suggest_int("epochs", 20, 80, step=10)
        batch = trial.suggest_categorical("batch_size", [16, 32, 64])

        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            # Create sequences
            X, y = [], []
            for i in range(seq_len, len(scaled)):
                X.append(scaled[i - seq_len:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            split = len(train_demand) - seq_len
            X_train, X_test = X[:split], X[split:]
            y_train_s, y_test_s = y[:split], y[split:]

            model = Sequential([
                LSTM(units, input_shape=(seq_len, 1), return_sequences=True),
                Dropout(dropout),
                LSTM(units // 2),
                Dropout(dropout),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train_s, epochs=epochs, batch_size=batch, verbose=0)

            preds_scaled = model.predict(X_test, verbose=0).flatten()
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            actual = scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()
            rmse = np.sqrt(np.mean((actual - preds) ** 2))
            return rmse
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize", study_name=f"lstm_{product}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n🏆 Best LSTM params for {product}:")
    print(f"   RMSE: {study.best_value:.2f}")
    for k, v in best.items():
        print(f"   {k}: {v}")

    import joblib
    param_path = os.path.join(config.RESULTS_DIR, f"best_lstm_params_{product}.pkl")
    joblib.dump(best, param_path)
    print(f"   Saved → {param_path}")

    return best, study.best_value


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not installed. Run: pip install optuna")
        return

    parser = argparse.ArgumentParser(description="Hyperparameter Tuner")
    parser.add_argument("--model", choices=["xgboost", "lstm", "all"], default="xgboost")
    parser.add_argument("--product", default=None, help="Product name (default: all)")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    products = [args.product] if args.product else config.PRODUCTS

    for product in products:
        print(f"\n{'='*60}")
        print(f"  Tuning {args.model.upper()} for: {product}")
        print(f"{'='*60}")

        if args.model in ("xgboost", "all"):
            tune_xgboost(product, n_trials=args.trials)
        if args.model in ("lstm", "all"):
            tune_lstm(product, n_trials=min(args.trials, 20))


if __name__ == "__main__":
    main()
