"""
Synthetic Demand Data Generator
Generates realistic retail demand data with seasonal patterns, promotions,
day-of-week effects, and trends across multiple product categories.
"""
import os
import numpy as np
import pandas as pd
from datetime import timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_demand_data(save=True):
    """Generate synthetic demand data for all products."""
    np.random.seed(config.RANDOM_STATE)
    dates = pd.date_range(start=config.DATA_START_DATE, end=config.DATA_END_DATE, freq="D")
    all_data = []

    for product in config.PRODUCTS:
        data = _generate_product_demand(product, dates)
        all_data.append(data)

    df = pd.concat(all_data, ignore_index=True)

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.GENERATED_DATA_FILE, index=False)
        print(f"✅ Generated {len(df):,} rows → {config.GENERATED_DATA_FILE}")

    return df


def _generate_product_demand(product, dates):
    """Generate demand for a single product category."""
    n = len(dates)

    # Base demand varies by product
    base_demands = {
        "Electronics": 150,
        "Clothing": 200,
        "Groceries": 500,
        "Furniture": 50,
        "Sports Equipment": 80,
    }
    base = base_demands.get(product, 100)

    # 1. Trend: gradual growth
    trend = np.linspace(0, base * 0.15, n)

    # 2. Yearly seasonality
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    yearly_season = _get_yearly_seasonality(product, day_of_year)

    # 3. Weekly seasonality (weekday effects)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_season = _get_weekly_seasonality(product, day_of_week)

    # 4. Holiday spikes
    holiday_effect = _get_holiday_effects(dates, base)

    # 5. Promotions (random events)
    promo_flag, promo_effect = _generate_promotions(n, base)

    # 6. Weather impact (seasonal proxy)
    weather_impact = 10 * np.sin(2 * np.pi * day_of_year / 365 + np.random.uniform(0, 2))

    # 7. Noise
    noise = np.random.normal(0, base * 0.08, n)

    # Combine all components
    demand = base + trend + yearly_season + weekly_season + holiday_effect + promo_effect + weather_impact + noise
    demand = np.maximum(demand, 0).astype(int)  # no negative demand

    # Unit price varies by product
    prices = {
        "Electronics": 299.99,
        "Clothing": 49.99,
        "Groceries": 12.99,
        "Furniture": 599.99,
        "Sports Equipment": 89.99,
    }

    df = pd.DataFrame({
        "date": dates,
        "product": product,
        "demand": demand,
        "price": prices.get(product, 50.0),
        "promotion": promo_flag,
        "day_of_week": day_of_week,
        "month": [d.month for d in dates],
        "is_weekend": (day_of_week >= 5).astype(int),
    })

    return df


def _get_yearly_seasonality(product, day_of_year):
    """Product-specific yearly seasonality."""
    seasonality_profiles = {
        "Electronics": 40 * np.sin(2 * np.pi * (day_of_year - 330) / 365),     # peaks near holidays
        "Clothing": 30 * np.sin(2 * np.pi * (day_of_year - 90) / 365),         # spring/fall peaks
        "Groceries": 15 * np.sin(2 * np.pi * (day_of_year - 350) / 365),       # slight holiday peak
        "Furniture": 20 * np.sin(2 * np.pi * (day_of_year - 180) / 365),       # mid-year peak
        "Sports Equipment": 35 * np.sin(2 * np.pi * (day_of_year - 150) / 365),# summer peak
    }
    return seasonality_profiles.get(product, 10 * np.sin(2 * np.pi * day_of_year / 365))


def _get_weekly_seasonality(product, day_of_week):
    """Day-of-week demand multipliers."""
    # Mon=0, Sun=6
    weekly_patterns = {
        "Electronics": np.array([-5, -3, 0, 5, 10, 20, 15]),
        "Clothing": np.array([-10, -5, 0, 5, 15, 25, 20]),
        "Groceries": np.array([10, 5, 0, 5, 20, 40, 30]),
        "Furniture": np.array([-5, -5, -3, 0, 5, 15, 10]),
        "Sports Equipment": np.array([-5, -5, -3, 0, 10, 20, 15]),
    }
    pattern = weekly_patterns.get(product, np.zeros(7))
    return pattern[day_of_week]


def _get_holiday_effects(dates, base):
    """Generate demand spikes around major holidays."""
    effects = np.zeros(len(dates))

    for i, d in enumerate(dates):
        # Christmas season (Dec 15 - Dec 25)
        if d.month == 12 and 15 <= d.day <= 25:
            effects[i] += base * 0.5
        # Black Friday (late November)
        elif d.month == 11 and 24 <= d.day <= 30:
            effects[i] += base * 0.6
        # New Year
        elif d.month == 1 and d.day <= 3:
            effects[i] += base * 0.2
        # Valentine's Day
        elif d.month == 2 and 12 <= d.day <= 14:
            effects[i] += base * 0.15
        # Back to school (August)
        elif d.month == 8 and 15 <= d.day <= 31:
            effects[i] += base * 0.1

    return effects


def _generate_promotions(n, base):
    """Generate random promotional events."""
    promo_flag = np.zeros(n, dtype=int)
    promo_effect = np.zeros(n)

    # Random promotion periods (roughly 10% of days)
    promo_starts = np.random.choice(range(n - 7), size=n // 30, replace=False)
    for start in promo_starts:
        duration = np.random.randint(3, 8)
        end = min(start + duration, n)
        promo_flag[start:end] = 1
        promo_effect[start:end] = base * np.random.uniform(0.1, 0.3)

    return promo_flag, promo_effect


if __name__ == "__main__":
    df = generate_demand_data()
    print(f"\nData shape: {df.shape}")
    print(f"Products: {df['product'].unique()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"\nSample:\n{df.head(10)}")
