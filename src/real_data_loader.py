"""
Real Data Loader — UCI Online Retail II Dataset
Downloads and transforms real e-commerce transaction data into the format
expected by the existing preprocessing and model training pipeline.

Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- ~1M transactions from a UK online retailer (2009-2011)
- Contains: InvoiceDate, Quantity, Price, Description, Country
"""
import os
import sys
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Download ─────────────────────────────────────────────────────────────────

def download_dataset():
    """Download the UCI Online Retail II Excel file if not already cached."""
    raw_path = os.path.join(config.DATA_DIR, "online_retail_II.xlsx")

    if os.path.exists(raw_path):
        print(f"  📁 Raw data already exists: {raw_path}")
        return raw_path

    print(f"  ⬇️  Downloading UCI Online Retail II dataset (~44 MB)...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    url = config.REAL_DATA_URL
    urllib.request.urlretrieve(url, raw_path)
    print(f"  ✅ Downloaded → {raw_path}")
    return raw_path


# ─── Product Categorization ──────────────────────────────────────────────────

# Keywords used to classify ~4000 product descriptions into 5 categories
CATEGORY_KEYWORDS = {
    "Home & Living": [
        "candle", "cushion", "frame", "clock", "lamp", "mirror", "vase",
        "decoration", "ornament", "pillow", "curtain", "rug", "blanket",
        "throw", "light", "lantern", "holder", "wall", "shelf", "hanging",
        "doormat", "coaster", "plaque", "sign", "hook", "drawer",
    ],
    "Kitchen & Dining": [
        "mug", "cup", "plate", "bowl", "napkin", "tray", "bottle",
        "kitchen", "cook", "bake", "spoon", "fork", "knife", "glass",
        "dish", "food", "lunch", "tea", "coffee", "cake", "dinner",
        "breakfast", "egg", "salt", "pepper", "sugar", "jam", "jug",
    ],
    "Gifts & Party": [
        "gift", "card", "party", "christmas", "birthday", "wedding",
        "wrapping", "ribbon", "bow", "box", "tag", "sticker", "star",
        "heart", "angel", "santa", "snowman", "garland", "tree", "bell",
        "tinsel", "bauble", "fairy", "bunting",
    ],
    "Garden & Outdoor": [
        "garden", "plant", "flower", "pot", "bird", "seed", "outdoor",
        "watering", "solar", "fence", "bench", "parasol", "butterfly",
        "insect", "herb", "grow", "nature", "wood", "stone", "metal",
    ],
    "Accessories & Fashion": [
        "bag", "purse", "wallet", "scarf", "jewel", "necklace", "bracelet",
        "ring", "earring", "charm", "bead", "pendant", "pin", "brooch",
        "hair", "hat", "glove", "socks", "key", "phone", "case", "watch",
    ],
}


def categorize_product(description):
    """Classify a product description into one of 5 categories."""
    if not isinstance(description, str):
        return None

    desc_lower = description.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category

    return None  # unclassifiable


# ─── Data Transformation ─────────────────────────────────────────────────────

def load_and_transform(raw_path):
    """
    Load raw Excel data and transform into the pipeline-expected format:
    date, product, demand, price, promotion, day_of_week, month, is_weekend
    """
    print("  📖 Reading Excel file (this may take a moment)...")

    # Read both sheets (Year 2009-2010 and Year 2010-2011)
    dfs = []
    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        try:
            df_sheet = pd.read_excel(raw_path, sheet_name=sheet, engine="openpyxl")
            dfs.append(df_sheet)
            print(f"    ✅ Loaded sheet '{sheet}': {len(df_sheet):,} rows")
        except Exception as e:
            print(f"    ⚠️ Could not load sheet '{sheet}': {e}")

    if not dfs:
        raise ValueError("No data sheets found in the Excel file")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  📊 Raw data: {len(df):,} total transactions")

    # ── Clean data ────────────────────────────────────────────────────────
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Rename columns for consistency
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "invoice" in cl and "date" in cl:
            col_map[col] = "InvoiceDate"
        elif "quantity" in cl:
            col_map[col] = "Quantity"
        elif "price" in cl and "stock" not in cl:
            col_map[col] = "Price"
        elif "description" in cl:
            col_map[col] = "Description"
        elif "stock" in cl and "code" in cl:
            col_map[col] = "StockCode"
        elif "invoice" in cl and "date" not in cl:
            col_map[col] = "Invoice"
    df.rename(columns=col_map, inplace=True)

    # Remove cancellations (invoices starting with 'C')
    if "Invoice" in df.columns:
        df["Invoice"] = df["Invoice"].astype(str)
        df = df[~df["Invoice"].str.startswith("C", na=False)]

    # Remove negative/zero quantities and prices
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Drop rows with missing descriptions
    df = df.dropna(subset=["Description", "InvoiceDate"])

    print(f"  🧹 After cleaning: {len(df):,} valid transactions")

    # ── Categorize products ───────────────────────────────────────────────
    df["category"] = df["Description"].apply(categorize_product)

    # Keep only categorized products
    df = df[df["category"].notna()].copy()
    print(f"  📦 Categorized: {len(df):,} transactions across {df['category'].nunique()} categories")

    for cat in sorted(df["category"].unique()):
        count = len(df[df["category"] == cat])
        print(f"    • {cat}: {count:,} transactions")

    # ── Aggregate to daily demand per category ────────────────────────────
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["date", "category"])
        .agg(
            demand=("Quantity", "sum"),
            price=("Price", "mean"),
        )
        .reset_index()
    )

    # ── Fill missing dates (some days have no transactions) ───────────────
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    categories = daily["category"].unique()

    full_index = pd.MultiIndex.from_product(
        [date_range, categories], names=["date", "category"]
    )
    daily = daily.set_index(["date", "category"]).reindex(full_index).reset_index()

    # Fill missing demand with 0, forward-fill prices
    daily["demand"] = daily["demand"].fillna(0).astype(int)
    daily["price"] = daily.groupby("category")["price"].transform(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    # ── Add temporal features ─────────────────────────────────────────────
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    # ── Generate promotion flag ───────────────────────────────────────────
    # Mark days with demand > 2 standard deviations above mean as promotions
    daily["promotion"] = 0
    for cat in categories:
        mask = daily["category"] == cat
        cat_demand = daily.loc[mask, "demand"]
        threshold = cat_demand.mean() + 2 * cat_demand.std()
        daily.loc[mask & (daily["demand"] > threshold), "promotion"] = 1

    # ── Rename columns to match pipeline format ───────────────────────────
    daily.rename(columns={"category": "product"}, inplace=True)

    # Sort
    daily.sort_values(["product", "date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    # Round price to 2 decimal places
    daily["price"] = daily["price"].round(2)

    return daily


# ─── Main Entry Point ────────────────────────────────────────────────────────

def load_real_data(save=True):
    """
    Full pipeline: download → clean → transform → save.
    Returns DataFrame in the same format as generate_demand_data().
    """
    print("\n📊 Loading Real Data (UCI Online Retail II)")
    print("=" * 50)

    raw_path = download_dataset()
    daily = load_and_transform(raw_path)

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        daily.to_csv(config.GENERATED_DATA_FILE, index=False)
        print(f"\n✅ Saved {len(daily):,} rows → {config.GENERATED_DATA_FILE}")

    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"  Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
    print(f"  Products: {daily['product'].nunique()}")
    print(f"  Total rows: {len(daily):,}")
    for p in sorted(daily["product"].unique()):
        avg = daily[daily["product"] == p]["demand"].mean()
        print(f"    • {p}: avg daily demand = {avg:.0f}")

    return daily


if __name__ == "__main__":
    df = load_real_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head(10)}")
