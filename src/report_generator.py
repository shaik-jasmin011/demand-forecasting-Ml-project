"""
Report Generator — Creates an HTML report with embedded charts and metrics.
Can be downloaded directly from the Streamlit dashboard.
"""
import os
import sys
import base64
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _encode_plotly_chart(fig):
    """Convert a Plotly figure to a base64 PNG image for embedding in HTML."""
    try:
        img_bytes = fig.to_image(format="png", width=900, height=400, scale=2)
        return base64.b64encode(img_bytes).decode()
    except Exception:
        # kaleido may not be installed; return empty
        return None


def _build_metrics_table(comparison_df):
    """Build an HTML table from the model comparison DataFrame."""
    if comparison_df is None or comparison_df.empty:
        return "<p>No model comparison data available.</p>"

    html = """
    <table style="width:100%; border-collapse:collapse; margin:20px 0;">
        <thead>
            <tr style="background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white;">
                <th style="padding:12px; text-align:left;">Model</th>
                <th style="padding:12px; text-align:right;">MAE</th>
                <th style="padding:12px; text-align:right;">RMSE</th>
                <th style="padding:12px; text-align:right;">MAPE (%)</th>
                <th style="padding:12px; text-align:right;">R²</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in comparison_df.iterrows():
        bg = "#f8f9fa" if i % 2 == 0 else "white"
        best_marker = " 🏆" if i == 0 else ""
        html += f"""
            <tr style="background:{bg};">
                <td style="padding:10px; font-weight:600;">{row['Model']}{best_marker}</td>
                <td style="padding:10px; text-align:right;">{row['MAE']:.2f}</td>
                <td style="padding:10px; text-align:right;">{row['RMSE']:.2f}</td>
                <td style="padding:10px; text-align:right;">{row['MAPE (%)']:.2f}</td>
                <td style="padding:10px; text-align:right;">{row['R²']:.4f}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html


def _build_inventory_section(inv_df):
    """Build HTML section for inventory optimization results."""
    if inv_df is None or inv_df.empty:
        return ""

    html = "<h2 style='color:#6366f1;'>📦 Inventory Optimization</h2>"
    html += """
    <table style="width:100%; border-collapse:collapse; margin:20px 0;">
        <thead>
            <tr style="background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white;">
                <th style="padding:10px; text-align:left;">Product</th>
                <th style="padding:10px; text-align:right;">EOQ</th>
                <th style="padding:10px; text-align:right;">Safety Stock</th>
                <th style="padding:10px; text-align:right;">Reorder Point</th>
                <th style="padding:10px; text-align:right;">Annual Cost</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in inv_df.iterrows():
        bg = "#f8f9fa" if i % 2 == 0 else "white"
        html += f"""
            <tr style="background:{bg};">
                <td style="padding:8px;">{row['product']}</td>
                <td style="padding:8px; text-align:right;">{row['eoq']:,.0f}</td>
                <td style="padding:8px; text-align:right;">{row['safety_stock']:,.0f}</td>
                <td style="padding:8px; text-align:right;">{row['reorder_point']:,.0f}</td>
                <td style="padding:8px; text-align:right;">${row['total_annual_cost']:,.2f}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html


def _build_feature_importance_section(product):
    """Build HTML section for feature importance."""
    fi_path = os.path.join(config.RESULTS_DIR, f"feature_importance_{product}.csv")
    if not os.path.exists(fi_path):
        return ""

    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(10)
    html = f"<h2 style='color:#6366f1;'>🔍 Top 10 Feature Importance — {product}</h2>"
    html += """
    <table style="width:60%; border-collapse:collapse; margin:20px 0;">
        <thead>
            <tr style="background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white;">
                <th style="padding:10px; text-align:left;">Rank</th>
                <th style="padding:10px; text-align:left;">Feature</th>
                <th style="padding:10px; text-align:right;">Importance</th>
            </tr>
        </thead>
        <tbody>
    """
    for rank, (_, row) in enumerate(fi.iterrows(), 1):
        bg = "#f8f9fa" if rank % 2 == 0 else "white"
        bar_width = row["importance"] / fi["importance"].max() * 100
        html += f"""
            <tr style="background:{bg};">
                <td style="padding:8px; font-weight:600;">#{rank}</td>
                <td style="padding:8px;">{row['feature']}</td>
                <td style="padding:8px; text-align:right;">
                    <div style="display:flex; align-items:center; justify-content:flex-end; gap:8px;">
                        <div style="background:linear-gradient(90deg,#818cf8,#f472b6); height:12px; width:{bar_width}%;
                            border-radius:6px; min-width:4px;"></div>
                        <span>{row['importance']:.4f}</span>
                    </div>
                </td>
            </tr>
        """
    html += "</tbody></table>"
    return html


def generate_report(selected_product=None):
    """
    Generate a full HTML report covering all products or a specific product.

    Returns:
        tuple (html_string, filename)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_mode = "UCI Online Retail II (Real)" if config.USE_REAL_DATA else "Synthetic"

    # Load data
    demand_path = config.GENERATED_DATA_FILE
    demand_data = pd.read_csv(demand_path, parse_dates=["date"]) if os.path.exists(demand_path) else None

    inv_path = os.path.join(config.RESULTS_DIR, "inventory_optimization.csv")
    inv_data = pd.read_csv(inv_path) if os.path.exists(inv_path) else None

    products = config.PRODUCTS
    if selected_product:
        products = [selected_product]

    # ── Build HTML ────────────────────────────────────────────────
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Demand Forecasting Report — {now}</title>
        <style>
            body {{
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                max-width: 1000px; margin: 0 auto; padding: 40px;
                color: #1e293b; background: #f8fafc;
            }}
            .header {{
                background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
                color: white; padding: 40px; border-radius: 16px; margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{ font-size: 2rem; margin: 0; color: #e0e7ff; }}
            .header p {{ color: #a5b4fc; margin: 8px 0 0; }}
            .section {{ background: white; border-radius: 12px; padding: 24px;
                margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            h2 {{ margin-top: 0; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
                margin: 20px 0; }}
            .kpi {{ background: linear-gradient(135deg, #eef2ff, #ede9fe); padding: 20px;
                border-radius: 12px; text-align: center; }}
            .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: #4f46e5; }}
            .kpi-label {{ font-size: 0.85rem; color: #64748b; margin-top: 4px; }}
            .footer {{ text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 20px; }}
            table {{ font-size: 0.9rem; }}
            @media print {{ body {{ padding: 20px; }} .section {{ page-break-inside: avoid; }} }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Demand Forecasting Report</h1>
            <p>Generated: {now} | Data: {data_mode} | Products: {len(products)}</p>
        </div>
    """

    # Overview KPIs
    if demand_data is not None:
        total_rows = len(demand_data)
        date_min = demand_data["date"].min().strftime("%Y-%m-%d")
        date_max = demand_data["date"].max().strftime("%Y-%m-%d")
        total_demand = demand_data["demand"].sum()

        html += f"""
        <div class="section">
            <h2 style="color:#6366f1;">📋 Dataset Overview</h2>
            <div class="kpi-grid">
                <div class="kpi">
                    <div class="kpi-value">{total_rows:,}</div>
                    <div class="kpi-label">Total Records</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{len(config.PRODUCTS)}</div>
                    <div class="kpi-label">Product Categories</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{total_demand:,.0f}</div>
                    <div class="kpi-label">Total Demand</div>
                </div>
                <div class="kpi">
                    <div class="kpi-value">{date_min}<br>{date_max}</div>
                    <div class="kpi-label">Date Range</div>
                </div>
            </div>
        </div>
        """

    # Per-product model comparison
    for product in products:
        comp_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{product}.csv")
        comp_data = pd.read_csv(comp_path) if os.path.exists(comp_path) else None

        html += f"""
        <div class="section">
            <h2 style="color:#6366f1;">📈 Model Performance — {product}</h2>
            {_build_metrics_table(comp_data)}
        </div>
        """

        # Feature importance
        fi_html = _build_feature_importance_section(product)
        if fi_html:
            html += f'<div class="section">{fi_html}</div>'

    # Inventory optimization
    if inv_data is not None:
        html += f'<div class="section">{_build_inventory_section(inv_data)}</div>'

    html += f"""
        <div class="footer">
            Report generated by DemandAI Platform • {now}
        </div>
    </body>
    </html>
    """

    filename = f"demand_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return html, filename


def save_report(output_dir=None, selected_product=None):
    """Generate and save the report to disk."""
    if output_dir is None:
        output_dir = config.RESULTS_DIR

    html, filename = generate_report(selected_product)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report saved → {filepath}")
    return filepath


if __name__ == "__main__":
    save_report()
