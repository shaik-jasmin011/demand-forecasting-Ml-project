"""
Model Evaluation & Comparison Framework
Computes MAE, RMSE, MAPE, R² across all models and generates comparison reports.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate evaluation metrics for a single model.
    
    Returns:
        dict with MAE, RMSE, MAPE, R²
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R²": round(r2, 4),
    }


def compare_models(results_dict):
    """
    Compare multiple models side by side.
    
    Args:
        results_dict: dict of {model_name: {"y_true": array, "y_pred": array}}
    
    Returns:
        DataFrame with comparison metrics
    """
    all_metrics = []
    for model_name, data in results_dict.items():
        metrics = calculate_metrics(
            data["y_true"], data["y_pred"], model_name=model_name
        )
        all_metrics.append(metrics)

    comparison = pd.DataFrame(all_metrics)
    comparison = comparison.sort_values("RMSE")

    return comparison


def generate_evaluation_report(results_dict, product_name="All"):
    """
    Generate and save full evaluation report.
    
    Args:
        results_dict: dict of {model_name: {"y_true": array, "y_pred": array}}
        product_name: name of the product for the report
    
    Returns:
        comparison DataFrame
    """
    comparison = compare_models(results_dict)

    # Save to CSV
    report_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{product_name}.csv")
    comparison.to_csv(report_path, index=False)

    print(f"\n{'='*60}")
    print(f"📊 Model Comparison — {product_name}")
    print(f"{'='*60}")
    print(comparison.to_string(index=False))
    print(f"\n🏆 Best model: {comparison.iloc[0]['Model']} (lowest RMSE)")
    print(f"📁 Report saved → {report_path}")

    return comparison


def save_predictions(test_dates, results_dict, product_name="All"):
    """
    Save all model predictions to a CSV for dashboard use.
    
    Args:
        test_dates: array/series of test dates
        results_dict: dict of {model_name: {"y_true": array, "y_pred": array}}
        product_name: product name
    """
    df = pd.DataFrame({"date": test_dates[:len(list(results_dict.values())[0]["y_true"])]})
    df["actual"] = list(results_dict.values())[0]["y_true"][:len(df)]

    for model_name, data in results_dict.items():
        preds = data["y_pred"][:len(df)]
        df[model_name] = preds

    df["product"] = product_name

    pred_path = os.path.join(config.RESULTS_DIR, f"predictions_{product_name}.csv")
    df.to_csv(pred_path, index=False)
    print(f"📁 Predictions saved → {pred_path}")

    return df
