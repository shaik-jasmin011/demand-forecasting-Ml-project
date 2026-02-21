"""
Logistics Optimization Module
Uses demand forecasts to recommend shipment schedules, warehouse allocation,
route prioritization, and cost estimation.
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def plan_shipments(forecast_df, product_name="Product"):
    """
    Create shipment schedule based on predicted demand peaks.
    Ships are planned when cumulative demand since last shipment exceeds a threshold.
    
    Args:
        forecast_df: DataFrame with 'date' and 'predicted_demand'
        product_name: product name
    
    Returns:
        DataFrame with shipment schedule
    """
    df = forecast_df.copy()
    avg_demand = df["predicted_demand"].mean()
    shipment_threshold = avg_demand * config.LEAD_TIME_DAYS  # ship when accumulated demand hits this

    shipments = []
    cumulative = 0
    last_ship_date = df["date"].iloc[0]

    for _, row in df.iterrows():
        cumulative += row["predicted_demand"]
        if cumulative >= shipment_threshold:
            shipments.append({
                "product": product_name,
                "ship_date": row["date"],
                "delivery_date": row["date"] + pd.Timedelta(days=config.LEAD_TIME_DAYS),
                "quantity": round(cumulative),
                "days_since_last_shipment": (row["date"] - last_ship_date).days,
                "priority": "High" if cumulative > shipment_threshold * 1.5 else "Normal",
            })
            last_ship_date = row["date"]
            cumulative = 0

    # Final partial shipment
    if cumulative > 0:
        shipments.append({
            "product": product_name,
            "ship_date": df["date"].iloc[-1],
            "delivery_date": df["date"].iloc[-1] + pd.Timedelta(days=config.LEAD_TIME_DAYS),
            "quantity": round(cumulative),
            "days_since_last_shipment": (df["date"].iloc[-1] - last_ship_date).days,
            "priority": "Normal",
        })

    return pd.DataFrame(shipments)


def allocate_warehouses(product_demands):
    """
    Allocate products to warehouses based on demand volume and proximity.
    Uses a greedy allocation: closest warehouse with remaining capacity.
    
    Args:
        product_demands: dict of {product_name: total_predicted_demand}
    
    Returns:
        DataFrame with warehouse allocation
    """
    # Sort warehouses by distance (closest first)
    sorted_warehouses = sorted(
        config.WAREHOUSE_DISTANCES.items(), key=lambda x: x[1]
    )

    remaining_capacity = config.WAREHOUSE_CAPACITIES.copy()
    allocations = []

    # Sort products by demand (highest first for priority allocation)
    sorted_products = sorted(product_demands.items(), key=lambda x: x[1], reverse=True)

    for product, demand in sorted_products:
        allocated = 0
        for warehouse, distance in sorted_warehouses:
            if remaining_capacity[warehouse] <= 0:
                continue

            alloc_qty = min(demand - allocated, remaining_capacity[warehouse])
            if alloc_qty <= 0:
                continue

            shipping_cost = alloc_qty * distance * config.SHIPPING_COST_PER_UNIT_PER_KM

            allocations.append({
                "product": product,
                "warehouse": warehouse,
                "allocated_units": round(alloc_qty),
                "warehouse_distance_km": distance,
                "shipping_cost": round(shipping_cost, 2),
                "capacity_utilization": round(
                    (config.WAREHOUSE_CAPACITIES[warehouse] - remaining_capacity[warehouse] + alloc_qty)
                    / config.WAREHOUSE_CAPACITIES[warehouse] * 100, 1
                ),
            })

            remaining_capacity[warehouse] -= alloc_qty
            allocated += alloc_qty

            if allocated >= demand:
                break

    return pd.DataFrame(allocations)


def calculate_route_priority(product_demands, urgency_threshold=0.8):
    """
    Score and prioritize delivery routes based on demand urgency.
    
    Args:
        product_demands: dict of {product_name: predicted_demand_array}
        urgency_threshold: percentile threshold for high urgency
    
    Returns:
        DataFrame with route priorities
    """
    routes = []
    all_demands = []

    for product, demand in product_demands.items():
        avg = np.mean(demand)
        peak = np.max(demand)
        volatility = np.std(demand) / avg if avg > 0 else 0
        all_demands.append(avg)

        routes.append({
            "product": product,
            "avg_daily_demand": round(avg, 1),
            "peak_demand": round(peak, 1),
            "demand_volatility": round(volatility, 3),
        })

    df = pd.DataFrame(routes)

    # Calculate priority score (0-100)
    demand_pct = df["avg_daily_demand"].rank(pct=True) * 40
    volatility_pct = df["demand_volatility"].rank(pct=True) * 30
    peak_pct = df["peak_demand"].rank(pct=True) * 30
    df["priority_score"] = (demand_pct + volatility_pct + peak_pct).round(1)
    df["priority_level"] = df["priority_score"].apply(
        lambda x: "🔴 Critical" if x > 80 else ("🟡 High" if x > 50 else "🟢 Normal")
    )

    df = df.sort_values("priority_score", ascending=False)
    return df


def estimate_logistics_costs(product_demands):
    """
    Estimate total logistics costs across all products and warehouses.
    
    Args:
        product_demands: dict of {product_name: total_predicted_demand}
    
    Returns:
        dict with cost breakdown
    """
    allocation = allocate_warehouses(product_demands)

    total_shipping = allocation["shipping_cost"].sum()
    total_units = allocation["allocated_units"].sum()
    avg_cost_per_unit = total_shipping / total_units if total_units > 0 else 0

    cost_by_warehouse = allocation.groupby("warehouse").agg({
        "shipping_cost": "sum",
        "allocated_units": "sum",
    }).to_dict("index")

    return {
        "total_shipping_cost": round(total_shipping, 2),
        "total_units_shipped": round(total_units),
        "avg_cost_per_unit": round(avg_cost_per_unit, 4),
        "cost_by_warehouse": cost_by_warehouse,
        "allocation_details": allocation,
    }


def generate_logistics_report(product_demands, product_forecast_arrays):
    """
    Generate comprehensive logistics report.
    
    Args:
        product_demands: dict of {product_name: total_predicted_demand}
        product_forecast_arrays: dict of {product_name: predicted_demand_array}
    
    Returns:
        dict with all logistics data
    """
    allocation = allocate_warehouses(product_demands)
    route_priority = calculate_route_priority(product_forecast_arrays)
    costs = estimate_logistics_costs(product_demands)

    # Save reports
    allocation.to_csv(os.path.join(config.RESULTS_DIR, "warehouse_allocation.csv"), index=False)
    route_priority.to_csv(os.path.join(config.RESULTS_DIR, "route_priority.csv"), index=False)

    print(f"\n🚚 Logistics reports saved → {config.RESULTS_DIR}")

    return {
        "allocation": allocation,
        "route_priority": route_priority,
        "costs": costs,
    }
