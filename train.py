"""
Main Training Pipeline
Orchestrates: data generation → preprocessing → model training → evaluation → optimization
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_generator import generate_demand_data
from src.real_data_loader import load_real_data
from src.preprocessing import load_data, add_features, split_data, get_feature_columns
from src.evaluation import calculate_metrics, generate_evaluation_report, save_predictions
from src.inventory_optimizer import optimize_all_products
from src.logistics_optimizer import generate_logistics_report

warnings.filterwarnings("ignore")


def train_all_models(product_name):
    """Train all 4 models for a given product and return results."""
    from src.models.arima_model import train_arima, predict_arima
    from src.models.prophet_model import train_prophet, predict_prophet
    from src.models.xgboost_model import train_xgboost, predict_xgboost, get_feature_importance
    from src.models.lstm_model import train_lstm, predict_lstm

    print(f"\n{'='*60}")
    print(f"🔄 Training models for: {product_name}")
    print(f"{'='*60}")

    # Load and preprocess
    df = load_data()
    df = add_features(df)
    train, test = split_data(df, product=product_name)
    feature_cols = get_feature_columns()

    y_true = test["demand"].values
    results = {}

    # ── 1. ARIMA ──────────────────────────────────────────────
    print("\n📈 Training ARIMA...")
    t0 = time.time()
    try:
        arima_model = train_arima(train, product_name=product_name)
        arima_preds = predict_arima(arima_model, steps=len(test))
        results["ARIMA"] = {"y_true": y_true, "y_pred": arima_preds}
        print(f"  ✅ ARIMA done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ ARIMA failed: {e}")

    # ── 2. Prophet ────────────────────────────────────────────
    print("\n📈 Training Prophet...")
    t0 = time.time()
    try:
        prophet_model = train_prophet(train, product_name=product_name)
        prophet_preds, _ = predict_prophet(prophet_model, test_data=test)
        results["Prophet"] = {"y_true": y_true, "y_pred": prophet_preds}
        print(f"  ✅ Prophet done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ Prophet failed: {e}")

    # ── 3. XGBoost ────────────────────────────────────────────
    print("\n📈 Training XGBoost...")
    t0 = time.time()
    try:
        xgb_model = train_xgboost(train, feature_cols, product_name=product_name)
        xgb_preds = predict_xgboost(xgb_model, test, feature_cols)
        results["XGBoost"] = {"y_true": y_true, "y_pred": xgb_preds}
        # Save feature importance
        importance_df = get_feature_importance(xgb_model, feature_cols)
        importance_df.to_csv(
            os.path.join(config.RESULTS_DIR, f"feature_importance_{product_name}.csv"),
            index=False,
        )
        print(f"  ✅ XGBoost done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ XGBoost failed: {e}")

    # ── 4. LSTM ───────────────────────────────────────────────
    print("\n📈 Training LSTM...")
    t0 = time.time()
    try:
        lstm_model, lstm_scaler = train_lstm(train, product_name=product_name)
        # For LSTM, we need the full series (train + test) to create sequences
        full_data = pd.concat([train, test], ignore_index=True)
        lstm_all_preds = predict_lstm(lstm_model, lstm_scaler, full_data)
        # Take only the test period predictions
        lstm_preds = lstm_all_preds[-(len(test)):]
        if len(lstm_preds) < len(y_true):
            lstm_preds = np.pad(lstm_preds, (len(y_true) - len(lstm_preds), 0), mode="edge")
        results["LSTM"] = {"y_true": y_true, "y_pred": lstm_preds[:len(y_true)]}
        print(f"  ✅ LSTM done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ LSTM failed: {e}")

    # ── 5. Ensemble (XGBoost + Prophet) ─────────────────────
    if "XGBoost" in results and "Prophet" in results:
        print("\n📈 Training Ensemble (XGBoost + Prophet)...")
        t0 = time.time()
        try:
            from src.models.ensemble_model import train_ensemble
            ens_result = train_ensemble(
                y_true,
                results["XGBoost"]["y_pred"],
                results["Prophet"]["y_pred"],
                product_name=product_name,
            )
            results["Ensemble"] = {"y_true": y_true, "y_pred": ens_result["predictions"]}
            print(f"  ✅ Ensemble done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ⚠️ Ensemble failed: {e}")

    return results, test


def main():
    """Run the full pipeline."""
    total_start = time.time()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   DEMAND FORECASTING & OPTIMIZATION PLATFORM        ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Step 1: Get Data ─────────────────────────────────────────
    if config.USE_REAL_DATA:
        print("\n📊 Step 1: Loading REAL data (UCI Online Retail II)...")
        load_real_data()
    else:
        print("\n📊 Step 1: Generating synthetic demand data...")
        generate_demand_data()

    # ── Step 2-4: Train, Evaluate, Save for each product ──────
    all_comparisons = []
    product_predictions = {}  # for optimization modules

    for product in config.PRODUCTS:
        results, test_data = train_all_models(product)

        if results:
            # Evaluate
            comparison = generate_evaluation_report(results, product_name=product)
            all_comparisons.append(comparison)

            # Save predictions
            save_predictions(test_data["date"].values, results, product_name=product)

            # Get best model predictions for optimization
            best_model = comparison.iloc[0]["Model"]
            product_predictions[product] = results[best_model]["y_pred"]

    # ── Step 5: Inventory Optimization ────────────────────────
    print("\n\n📦 Step 5: Running Inventory Optimization...")
    inventory_results = optimize_all_products(product_predictions)
    print(inventory_results.to_string(index=False))

    # ── Step 6: Logistics Optimization ────────────────────────
    print("\n🚚 Step 6: Running Logistics Optimization...")
    total_demands = {p: np.sum(d) for p, d in product_predictions.items()}
    logistics = generate_logistics_report(total_demands, product_predictions)

    print("\n📍 Warehouse Allocation:")
    print(logistics["allocation"].to_string(index=False))
    print("\n📍 Route Priority:")
    print(logistics["route_priority"].to_string(index=False))

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE in {total_time:.1f} seconds")
    print(f"📁 Results saved to: {config.RESULTS_DIR}")
    print(f"📁 Models saved to: {config.MODEL_DIR}")
    print(f"{'='*60}")

    # Save combined comparison
    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        combined.to_csv(os.path.join(config.RESULTS_DIR, "all_model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
