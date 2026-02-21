"""
Live Data Simulator — Generates realistic new daily demand records.
Simulates real-time data arrival by appending new rows to the existing
demand_data.csv file, mimicking a live e-commerce data feed.

Usage:
    python -m src.live_simulator              # add 1 day
    python -m src.live_simulator --days 7     # add 7 days
    python -m src.live_simulator --continuous # run forever, 1 day every N seconds
"""
import os
import sys
import time
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _seasonal_demand(date, base, product_idx):
    """Generate demand with realistic seasonal patterns."""
    day_of_week = date.dayofweek
    month = date.month

    # Weekend boost
    weekend_factor = 1.25 if day_of_week >= 5 else 1.0

    # Monthly seasonality (holiday peaks in Nov-Dec, dip in Jan-Feb)
    month_factors = {
        1: 0.7, 2: 0.75, 3: 0.85, 4: 0.9, 5: 0.95, 6: 1.0,
        7: 1.0, 8: 0.95, 9: 1.05, 10: 1.1, 11: 1.3, 12: 1.5,
    }
    month_factor = month_factors.get(month, 1.0)

    # Weekly cycle (mid-week dip)
    day_factors = [0.95, 0.9, 0.88, 0.92, 1.05, 1.15, 1.1]
    day_factor = day_factors[day_of_week]

    # Random noise
    noise = np.random.normal(1.0, 0.15)

    # Trend (slight upward)
    days_since_start = (date - pd.Timestamp("2009-01-01")).days
    trend = 1.0 + days_since_start * 0.0001

    demand = base * weekend_factor * month_factor * day_factor * noise * trend
    return max(int(demand), 1)


def _generate_day(date, products, product_stats):
    """Generate demand records for all products for a single day."""
    rows = []
    for i, product in enumerate(products):
        stats = product_stats.get(product, {"mean_demand": 100, "mean_price": 10.0})

        demand = _seasonal_demand(date, stats["mean_demand"], i)
        price = round(stats["mean_price"] * np.random.uniform(0.9, 1.1), 2)
        promotion = 1 if np.random.random() < 0.12 else 0

        if promotion:
            demand = int(demand * np.random.uniform(1.5, 2.5))

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product": product,
            "demand": demand,
            "price": price,
            "promotion": promotion,
            "day_of_week": date.dayofweek,
            "month": date.month,
            "is_weekend": int(date.dayofweek >= 5),
        })
    return rows


def get_product_stats(df):
    """Calculate baseline stats from existing data for realistic simulation."""
    stats = {}
    for product in df["product"].unique():
        pdata = df[df["product"] == product]
        stats[product] = {
            "mean_demand": pdata["demand"].mean(),
            "mean_price": pdata["price"].mean(),
        }
    return stats


def simulate_new_data(num_days=1):
    """
    Append num_days of new simulated demand data to the existing CSV.

    Returns:
        tuple (num_rows_added, new_last_date)
    """
    csv_path = config.GENERATED_DATA_FILE

    if not os.path.exists(csv_path):
        print("❌ No existing data found. Run `python train.py` first.")
        return 0, None

    df = pd.read_csv(csv_path, parse_dates=["date"])
    last_date = df["date"].max()
    products = df["product"].unique().tolist()
    product_stats = get_product_stats(df)

    new_rows = []
    for d in range(1, num_days + 1):
        new_date = last_date + timedelta(days=d)
        new_rows.extend(_generate_day(new_date, products, product_stats))

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    new_last = (last_date + timedelta(days=num_days)).strftime("%Y-%m-%d")
    print(f"✅ Added {len(new_rows)} records ({num_days} days × {len(products)} products)")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} → {new_last}")
    print(f"   Total records: {len(combined):,}")

    return len(new_rows), new_last


def run_continuous(interval_seconds=30):
    """Continuously generate 1 new day every N seconds."""
    print(f"🔄 Live mode: generating 1 day every {interval_seconds}s (Ctrl+C to stop)")
    print("="*60)

    while True:
        simulate_new_data(num_days=1)

        # Touch a flag file so the scheduler knows new data arrived
        flag_path = os.path.join(config.DATA_DIR, ".new_data_flag")
        with open(flag_path, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

        print(f"   ⏳ Next update in {interval_seconds}s...\n")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Data Simulator")
    parser.add_argument("--days", type=int, default=1, help="Number of days to simulate")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between updates (continuous mode)")
    args = parser.parse_args()

    if args.continuous:
        run_continuous(interval_seconds=args.interval)
    else:
        simulate_new_data(num_days=args.days)
