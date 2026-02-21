"""
Inventory Optimization Module
Uses demand forecasts to calculate EOQ, safety stock, reorder points,
and ABC classification for inventory management.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def calculate_eoq(annual_demand, ordering_cost=None, holding_cost=None):
    """
    Economic Order Quantity (EOQ).
    
    EOQ = sqrt(2 * D * S / H)
    where D = annual demand, S = ordering cost, H = holding cost per unit
    """
    if ordering_cost is None:
        ordering_cost = config.ORDERING_COST_PER_ORDER
    if holding_cost is None:
        holding_cost = config.HOLDING_COST_PER_UNIT

    if holding_cost <= 0 or annual_demand <= 0:
        return 0

    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
    return round(eoq)


def calculate_safety_stock(demand_std, lead_time=None, service_level=None):
    """
    Safety Stock = Z * σ_d * √L
    where Z = z-score for service level, σ_d = daily demand std, L = lead time
    """
    if lead_time is None:
        lead_time = config.LEAD_TIME_DAYS
    if service_level is None:
        service_level = config.SERVICE_LEVEL

    z_score = stats.norm.ppf(service_level)
    safety = z_score * demand_std * np.sqrt(lead_time)
    return round(safety)


def calculate_reorder_point(avg_daily_demand, lead_time=None, safety_stock=0):
    """
    Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
    """
    if lead_time is None:
        lead_time = config.LEAD_TIME_DAYS

    rop = (avg_daily_demand * lead_time) + safety_stock
    return round(rop)


def abc_classification(products_data):
    """
    ABC Classification based on total demand value.
    A = top 80% of value (typically ~20% of items)
    B = next 15% of value
    C = remaining 5% of value
    """
    df = products_data.copy()
    df["total_value"] = df["total_demand"] * df["price"]
    df = df.sort_values("total_value", ascending=False)
    df["cumulative_pct"] = df["total_value"].cumsum() / df["total_value"].sum() * 100

    def classify(pct):
        if pct <= 80:
            return "A"
        elif pct <= 95:
            return "B"
        else:
            return "C"

    df["abc_class"] = df["cumulative_pct"].apply(classify)
    return df


def optimize_inventory(forecast_df, product_name="All"):
    """
    Run full inventory optimization for a product.
    
    Args:
        forecast_df: DataFrame with 'date' and 'predicted_demand' columns
        product_name: product name
    
    Returns:
        dict with all inventory optimization metrics
    """
    daily_demand = forecast_df["predicted_demand"].values
    avg_daily = np.mean(daily_demand)
    std_daily = np.std(daily_demand)
    annual_demand = avg_daily * 365

    eoq = calculate_eoq(annual_demand)
    safety = calculate_safety_stock(std_daily)
    rop = calculate_reorder_point(avg_daily, safety_stock=safety)

    # Number of orders per year
    orders_per_year = round(annual_demand / eoq) if eoq > 0 else 0

    # Total annual cost
    annual_ordering_cost = orders_per_year * config.ORDERING_COST_PER_ORDER
    annual_holding_cost = (eoq / 2) * config.HOLDING_COST_PER_UNIT
    total_annual_cost = annual_ordering_cost + annual_holding_cost

    # Days of supply
    days_of_supply = round(eoq / avg_daily) if avg_daily > 0 else 0

    result = {
        "product": product_name,
        "avg_daily_demand": round(avg_daily, 1),
        "demand_std": round(std_daily, 1),
        "annual_demand": round(annual_demand),
        "eoq": eoq,
        "safety_stock": safety,
        "reorder_point": rop,
        "orders_per_year": orders_per_year,
        "days_of_supply": days_of_supply,
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
        "lead_time_days": config.LEAD_TIME_DAYS,
        "service_level": config.SERVICE_LEVEL,
    }

    return result


def optimize_all_products(predictions_dict):
    """
    Run inventory optimization for all products.
    
    Args:
        predictions_dict: dict of {product_name: predicted_demand_array}
    
    Returns:
        DataFrame with optimization results
    """
    results = []
    for product, demand in predictions_dict.items():
        forecast_df = pd.DataFrame({"predicted_demand": demand})
        result = optimize_inventory(forecast_df, product_name=product)
        results.append(result)

    df = pd.DataFrame(results)

    # Save results
    path = os.path.join(config.RESULTS_DIR, "inventory_optimization.csv")
    df.to_csv(path, index=False)
    print(f"\n📦 Inventory optimization results saved → {path}")

    return df
