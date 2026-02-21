"""
Demand Forecasting & Optimization Dashboard
Premium Streamlit dashboard with 4 interactive pages:
1. 📈 Demand Forecast
2. 📦 Inventory Optimizer
3. 🚚 Logistics Planner
4. 📊 Model Comparison
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.inventory_optimizer import optimize_inventory, abc_classification
from src.logistics_optimizer import (
    plan_shipments, allocate_warehouses,
    calculate_route_priority, estimate_logistics_costs
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

div[data-testid="metric-container"] label {
    color: #a5b4fc !important;
    font-weight: 500;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e0e7ff !important;
    font-weight: 700;
}

/* Headers */
h1, h2, h3 {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30, 30, 60, 0.5);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #a5b4fc;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    overflow: hidden;
}

/* Select boxes */
.stSelectbox > div > div {
    background: rgba(30, 30, 60, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: rgba(99, 102, 241, 0.2) !important;
}

.glass-card {
    background: rgba(30, 30, 60, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.hero-title {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 300;
}

.kpi-label { color: #a5b4fc; font-size: 0.85rem; font-weight: 500; }
.kpi-value { color: #e0e7ff; font-size: 1.8rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_demand_data():
    path = config.GENERATED_DATA_FILE
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_data
def load_predictions(product):
    path = os.path.join(config.RESULTS_DIR, f"predictions_{product}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_data
def load_model_comparison():
    path = os.path.join(config.RESULTS_DIR, "all_model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_inventory_results():
    path = os.path.join(config.RESULTS_DIR, "inventory_optimization.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_importance(product):
    path = os.path.join(config.RESULTS_DIR, f"feature_importance_{product}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e0e7ff", "family": "Inter"},
        "xaxis": {
            "gridcolor": "rgba(99,102,241,0.1)",
            "linecolor": "rgba(99,102,241,0.3)",
        },
        "yaxis": {
            "gridcolor": "rgba(99,102,241,0.1)",
            "linecolor": "rgba(99,102,241,0.3)",
        },
        "colorway": ["#818cf8", "#c084fc", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"],
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#a5b4fc"}},
    }
}

COLORS = {
    "primary": "#818cf8",
    "secondary": "#c084fc",
    "accent": "#f472b6",
    "success": "#34d399",
    "warning": "#fbbf24",
    "info": "#60a5fa",
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="font-size: 1.5rem;">📊 DemandAI</h2>
        <p class="hero-subtitle">Forecasting & Optimization</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["📈 Demand Forecast", "📦 Inventory Optimizer", "🚚 Logistics Planner", "📊 Model Comparison", "🔍 Feature Importance"],
        label_visibility="collapsed",
    )

    st.divider()

    demand_data = load_demand_data()
    if demand_data is not None:
        products = demand_data["product"].unique().tolist()
        selected_product = st.selectbox("Select Product", products)
    else:
        st.error("⚠️ No data found. Run `python train.py` first.")
        st.stop()

    st.divider()

    # Report download
    if st.button("📄 Download Report", use_container_width=True):
        from src.report_generator import generate_report
        html, filename = generate_report(selected_product)
        st.download_button(
            label="💾 Save Report (HTML)",
            data=html,
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )

    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 10px; color: #64748b; font-size: 0.8rem;">
        Built with Streamlit & ML<br>
        © 2024 DemandAI Platform
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Demand Forecast
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📈 Demand Forecast":
    st.markdown("""
    <div class="glass-card">
        <p class="hero-title">📈 Demand Forecast</p>
        <p class="hero-subtitle">Historical trends and AI-powered future demand predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Product data
    product_data = demand_data[demand_data["product"] == selected_product].copy()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Daily Demand", f"{product_data['demand'].mean():.0f}")
    with col2:
        st.metric("Peak Demand", f"{product_data['demand'].max():.0f}")
    with col3:
        st.metric("Demand Std Dev", f"{product_data['demand'].std():.0f}")
    with col4:
        total = product_data["demand"].sum()
        st.metric("Total Demand", f"{total:,.0f}")

    st.markdown("---")

    # Historical demand chart
    tab1, tab2, tab3 = st.tabs(["📊 Time Series", "📅 Seasonal Pattern", "🔮 Forecast"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=product_data["date"],
            y=product_data["demand"],
            mode="lines",
            name="Actual Demand",
            line=dict(color=COLORS["primary"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(129, 140, 248, 0.1)",
        ))

        # Add rolling average
        product_data["rolling_30"] = product_data["demand"].rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=product_data["date"],
            y=product_data["rolling_30"],
            mode="lines",
            name="30-Day Moving Avg",
            line=dict(color=COLORS["accent"], width=2.5),
        ))

        fig.update_layout(
            title=f"Demand History — {selected_product}",
            xaxis_title="Date",
            yaxis_title="Daily Demand (units)",
            template=PLOTLY_TEMPLATE,
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Monthly seasonality
        product_data["month_name"] = product_data["date"].dt.month_name()
        monthly = product_data.groupby("month")["demand"].agg(["mean", "std"]).reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = [month_names[i-1] for i in monthly["month"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month_name"],
            y=monthly["mean"],
            marker=dict(
                color=monthly["mean"],
                colorscale=[[0, COLORS["info"]], [0.5, COLORS["primary"]], [1, COLORS["accent"]]],
                cornerradius=8,
            ),
            error_y=dict(type="data", array=monthly["std"], color="rgba(255,255,255,0.3)"),
            name="Avg Demand",
        ))
        fig.update_layout(
            title=f"Monthly Seasonal Pattern — {selected_product}",
            xaxis_title="Month",
            yaxis_title="Average Demand",
            template=PLOTLY_TEMPLATE,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Day of week pattern
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = product_data.groupby("day_of_week")["demand"].mean().reset_index()
        dow["day_name"] = [dow_names[i] for i in dow["day_of_week"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dow["day_name"],
            y=dow["demand"],
            marker=dict(
                color=[COLORS["primary"]] * 5 + [COLORS["success"]] * 2,
                cornerradius=8,
            ),
        ))
        fig.update_layout(
            title=f"Day-of-Week Pattern — {selected_product}",
            xaxis_title="Day",
            yaxis_title="Average Demand",
            template=PLOTLY_TEMPLATE,
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        preds = load_predictions(selected_product)
        if preds is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds["date"], y=preds["actual"],
                mode="lines", name="Actual",
                line=dict(color="white", width=2),
            ))

            model_colors = {
                "ARIMA": COLORS["primary"],
                "Prophet": COLORS["secondary"],
                "XGBoost": COLORS["success"],
                "LSTM": COLORS["accent"],
            }

            for model in ["ARIMA", "Prophet", "XGBoost", "LSTM"]:
                if model in preds.columns:
                    fig.add_trace(go.Scatter(
                        x=preds["date"], y=preds[model],
                        mode="lines", name=model,
                        line=dict(color=model_colors.get(model, COLORS["info"]), width=1.5, dash="dot"),
                    ))

            fig.update_layout(
                title=f"Forecast vs Actual — {selected_product}",
                xaxis_title="Date",
                yaxis_title="Demand",
                template=PLOTLY_TEMPLATE,
                height=500,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔮 Run `python train.py` to generate forecast predictions.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Inventory Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Optimizer":
    st.markdown("""
    <div class="glass-card">
        <p class="hero-title">📦 Inventory Optimizer</p>
        <p class="hero-subtitle">EOQ, safety stock, and reorder point analysis powered by demand forecasts</p>
    </div>
    """, unsafe_allow_html=True)

    inv_results = load_inventory_results()

    if inv_results is not None:
        # Product-level metrics
        product_inv = inv_results[inv_results["product"] == selected_product]

        if not product_inv.empty:
            row = product_inv.iloc[0]

            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("EOQ (units)", f"{row['eoq']:,.0f}")
            with col2:
                st.metric("Safety Stock", f"{row['safety_stock']:,.0f}")
            with col3:
                st.metric("Reorder Point", f"{row['reorder_point']:,.0f}")
            with col4:
                st.metric("Annual Cost", f"${row['total_annual_cost']:,.2f}")

            st.markdown("---")

            col_left, col_right = st.columns(2)

            with col_left:
                # Cost breakdown donut
                fig = go.Figure(data=[go.Pie(
                    labels=["Ordering Cost", "Holding Cost"],
                    values=[row["annual_ordering_cost"], row["annual_holding_cost"]],
                    hole=0.6,
                    marker=dict(colors=[COLORS["primary"], COLORS["accent"]]),
                    textinfo="label+percent",
                    textfont=dict(color="white"),
                )])
                fig.update_layout(
                    title=f"Annual Cost Breakdown — {selected_product}",
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    showlegend=False,
                    annotations=[{
                        "text": f"${row['total_annual_cost']:,.0f}",
                        "x": 0.5, "y": 0.5,
                        "font_size": 20, "font_color": "white",
                        "showarrow": False,
                    }],
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Inventory parameters table
                params = pd.DataFrame({
                    "Parameter": [
                        "Avg Daily Demand", "Demand Std Dev", "Annual Demand",
                        "Lead Time", "Service Level", "Days of Supply",
                        "Orders per Year",
                    ],
                    "Value": [
                        f"{row['avg_daily_demand']:.0f} units",
                        f"{row['demand_std']:.0f} units",
                        f"{row['annual_demand']:,.0f} units",
                        f"{row['lead_time_days']} days",
                        f"{row['service_level']*100:.0f}%",
                        f"{row['days_of_supply']} days",
                        f"{row['orders_per_year']}",
                    ],
                })
                st.markdown("### 📋 Inventory Parameters")
                st.dataframe(params, use_container_width=True, hide_index=True, height=300)

            st.markdown("---")

            # All products comparison
            st.markdown("### 🏷️ All Products Comparison")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=inv_results["product"], y=inv_results["eoq"],
                name="EOQ",
                marker=dict(color=COLORS["primary"], cornerradius=6),
            ))
            fig.add_trace(go.Bar(
                x=inv_results["product"], y=inv_results["safety_stock"],
                name="Safety Stock",
                marker=dict(color=COLORS["warning"], cornerradius=6),
            ))
            fig.add_trace(go.Bar(
                x=inv_results["product"], y=inv_results["reorder_point"],
                name="Reorder Point",
                marker=dict(color=COLORS["accent"], cornerradius=6),
            ))
            fig.update_layout(
                barmode="group",
                title="Inventory Levels by Product",
                template=PLOTLY_TEMPLATE,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ABC Classification
            st.markdown("### 🏅 ABC Classification")
            pdata = demand_data.groupby("product").agg(
                total_demand=("demand", "sum"),
                price=("price", "first"),
            ).reset_index()
            abc = abc_classification(pdata)
            abc_display = abc[["product", "total_demand", "price", "total_value", "cumulative_pct", "abc_class"]]
            abc_display.columns = ["Product", "Total Demand", "Unit Price", "Total Value ($)", "Cumulative %", "Class"]

            st.dataframe(abc_display, use_container_width=True, hide_index=True)

    else:
        st.info("📦 Run `python train.py` to generate inventory optimization results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Logistics Planner
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 Logistics Planner":
    st.markdown("""
    <div class="glass-card">
        <p class="hero-title">🚚 Logistics Planner</p>
        <p class="hero-subtitle">Shipment scheduling, warehouse allocation, and cost optimization</p>
    </div>
    """, unsafe_allow_html=True)

    preds = load_predictions(selected_product)

    if preds is not None:
        tab1, tab2, tab3 = st.tabs(["📦 Shipment Schedule", "🏭 Warehouse Allocation", "💰 Cost Analysis"])

        with tab1:
            # Get best model predictions
            model_cols = [c for c in preds.columns if c not in ["date", "actual", "product"]]
            best_model = model_cols[0] if model_cols else "actual"
            pred_col = best_model

            shipment_df = pd.DataFrame({
                "date": preds["date"],
                "predicted_demand": preds[pred_col],
            })
            schedule = plan_shipments(shipment_df, product_name=selected_product)

            if not schedule.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Shipments", len(schedule))
                with col2:
                    st.metric("Total Units", f"{schedule['quantity'].sum():,.0f}")
                with col3:
                    high_prio = len(schedule[schedule["priority"] == "High"])
                    st.metric("High Priority", high_prio)

                st.markdown("---")

                # Shipment timeline
                fig = go.Figure()
                colors = schedule["priority"].map({"High": COLORS["accent"], "Normal": COLORS["primary"]})
                fig.add_trace(go.Bar(
                    x=schedule["ship_date"],
                    y=schedule["quantity"],
                    marker=dict(color=colors, cornerradius=6),
                    text=schedule["priority"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Shipment Schedule — {selected_product}",
                    xaxis_title="Ship Date",
                    yaxis_title="Quantity",
                    template=PLOTLY_TEMPLATE,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Shipment Details")
                st.dataframe(schedule, use_container_width=True, hide_index=True)

        with tab2:
            # Warehouse allocation
            all_preds = {}
            for p in config.PRODUCTS:
                p_pred = load_predictions(p)
                if p_pred is not None:
                    model_cols = [c for c in p_pred.columns if c not in ["date", "actual", "product"]]
                    if model_cols:
                        all_preds[p] = p_pred[model_cols[0]].sum()

            if all_preds:
                allocation = allocate_warehouses(all_preds)

                if not allocation.empty:
                    # Warehouse utilization
                    util_data = allocation.groupby("warehouse").agg(
                        total_units=("allocated_units", "sum"),
                        total_cost=("shipping_cost", "sum"),
                    ).reset_index()

                    col1, col2 = st.columns(2)

                    with col1:
                        fig = go.Figure(data=[go.Pie(
                            labels=util_data["warehouse"],
                            values=util_data["total_units"],
                            hole=0.5,
                            marker=dict(colors=[COLORS["primary"], COLORS["secondary"], COLORS["success"]]),
                            textinfo="label+percent",
                            textfont=dict(color="white"),
                        )])
                        fig.update_layout(
                            title="Units by Warehouse",
                            template=PLOTLY_TEMPLATE,
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = go.Figure(data=[go.Bar(
                            x=util_data["warehouse"],
                            y=util_data["total_cost"],
                            marker=dict(
                                color=util_data["total_cost"],
                                colorscale=[[0, COLORS["success"]], [1, COLORS["accent"]]],
                                cornerradius=8,
                            ),
                            text=[f"${c:,.0f}" for c in util_data["total_cost"]],
                            textposition="outside",
                            textfont=dict(color="white"),
                        )])
                        fig.update_layout(
                            title="Shipping Cost by Warehouse",
                            yaxis_title="Cost ($)",
                            template=PLOTLY_TEMPLATE,
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📋 Allocation Details")
                    st.dataframe(allocation, use_container_width=True, hide_index=True)

        with tab3:
            # Cost analysis
            if all_preds:
                costs = estimate_logistics_costs(all_preds)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Shipping Cost", f"${costs['total_shipping_cost']:,.2f}")
                with col2:
                    st.metric("Total Units Shipped", f"{costs['total_units_shipped']:,.0f}")
                with col3:
                    st.metric("Avg Cost/Unit", f"${costs['avg_cost_per_unit']:.4f}")

                st.markdown("---")

                # Route priority
                pred_arrays = {}
                for p in config.PRODUCTS:
                    p_pred = load_predictions(p)
                    if p_pred is not None:
                        model_cols = [c for c in p_pred.columns if c not in ["date", "actual", "product"]]
                        if model_cols:
                            pred_arrays[p] = p_pred[model_cols[0]].values

                if pred_arrays:
                    route_df = calculate_route_priority(pred_arrays)
                    st.markdown("### 📍 Route Priority Scoring")
                    st.dataframe(route_df, use_container_width=True, hide_index=True)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=route_df["product"],
                        y=route_df["priority_score"],
                        marker=dict(
                            color=route_df["priority_score"],
                            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["accent"]]],
                            cornerradius=8,
                        ),
                        text=[f"{s:.0f}" for s in route_df["priority_score"]],
                        textposition="outside",
                        textfont=dict(color="white"),
                    ))
                    fig.update_layout(
                        title="Route Priority Scores",
                        yaxis_title="Priority Score",
                        template=PLOTLY_TEMPLATE,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("🚚 Run `python train.py` to generate logistics data.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("""
    <div class="glass-card">
        <p class="hero-title">📊 Model Comparison</p>
        <p class="hero-subtitle">Side-by-side accuracy metrics and residual analysis across all models</p>
    </div>
    """, unsafe_allow_html=True)

    comparison = load_model_comparison()

    if comparison is not None:
        # Filter for selected product (from per-product reports)
        product_comp_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{selected_product}.csv")
        if os.path.exists(product_comp_path):
            product_comp = pd.read_csv(product_comp_path)
        else:
            product_comp = comparison

        # Best model highlight
        best = product_comp.iloc[0]
        st.success(f"🏆 **Best Model: {best['Model']}** — RMSE: {best['RMSE']:.2f} | MAE: {best['MAE']:.2f} | R²: {best['R²']:.4f}")

        st.markdown("---")

        # Metrics comparison
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Predictions", "🔍 Residuals"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                # RMSE comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=product_comp["Model"],
                    y=product_comp["RMSE"],
                    marker=dict(
                        color=product_comp["RMSE"],
                        colorscale=[[0, COLORS["success"]], [1, COLORS["accent"]]],
                        cornerradius=8,
                    ),
                    text=[f"{v:.1f}" for v in product_comp["RMSE"]],
                    textposition="outside",
                    textfont=dict(color="white"),
                ))
                fig.update_layout(
                    title="RMSE (lower is better)",
                    template=PLOTLY_TEMPLATE,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # R² comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=product_comp["Model"],
                    y=product_comp["R²"],
                    marker=dict(
                        color=product_comp["R²"],
                        colorscale=[[0, COLORS["accent"]], [1, COLORS["success"]]],
                        cornerradius=8,
                    ),
                    text=[f"{v:.4f}" for v in product_comp["R²"]],
                    textposition="outside",
                    textfont=dict(color="white"),
                ))
                fig.update_layout(
                    title="R² Score (higher is better)",
                    template=PLOTLY_TEMPLATE,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Full metrics table
            st.markdown("### 📋 Full Metrics Table")
            st.dataframe(product_comp, use_container_width=True, hide_index=True)

            # Radar chart
            if len(product_comp) >= 2:
                categories = ["MAE", "RMSE", "MAPE (%)", "R²"]
                fig = go.Figure()

                for _, row in product_comp.iterrows():
                    # Normalize values for radar (0-1 scale)
                    max_mae = product_comp["MAE"].max()
                    max_rmse = product_comp["RMSE"].max()
                    max_mape = product_comp["MAPE (%)"].max()

                    values = [
                        1 - (row["MAE"] / max_mae if max_mae else 0),
                        1 - (row["RMSE"] / max_rmse if max_rmse else 0),
                        1 - (row["MAPE (%)"] / max_mape if max_mape else 0),
                        max(row["R²"], 0),
                    ]
                    values.append(values[0])  # close the polygon

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        name=row["Model"],
                        fill="toself",
                        opacity=0.3,
                    ))

                fig.update_layout(
                    title="Model Performance Radar",
                    template=PLOTLY_TEMPLATE,
                    height=450,
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(gridcolor="rgba(99,102,241,0.2)", linecolor="rgba(99,102,241,0.3)"),
                        angularaxis=dict(gridcolor="rgba(99,102,241,0.2)", linecolor="rgba(99,102,241,0.3)"),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            preds = load_predictions(selected_product)
            if preds is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=preds["date"], y=preds["actual"],
                    mode="lines", name="Actual",
                    line=dict(color="white", width=2.5),
                ))

                model_colors = {
                    "ARIMA": COLORS["primary"],
                    "Prophet": COLORS["secondary"],
                    "XGBoost": COLORS["success"],
                    "LSTM": COLORS["accent"],
                }

                for model in ["ARIMA", "Prophet", "XGBoost", "LSTM"]:
                    if model in preds.columns:
                        fig.add_trace(go.Scatter(
                            x=preds["date"], y=preds[model],
                            mode="lines", name=model,
                            line=dict(color=model_colors.get(model), width=1.5),
                        ))

                fig.update_layout(
                    title=f"All Models vs Actual — {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Demand",
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            preds = load_predictions(selected_product)
            if preds is not None:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=["ARIMA", "Prophet", "XGBoost", "LSTM"],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08,
                )

                positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                model_colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["accent"]]

                for idx, model in enumerate(["ARIMA", "Prophet", "XGBoost", "LSTM"]):
                    if model in preds.columns:
                        residuals = preds["actual"] - preds[model]
                        r, c = positions[idx]
                        fig.add_trace(go.Scatter(
                            x=preds["date"], y=residuals,
                            mode="lines", name=f"{model} Residuals",
                            line=dict(color=model_colors_list[idx], width=1),
                            showlegend=False,
                        ), row=r, col=c)
                        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=r, col=c)

                fig.update_layout(
                    title=f"Residual Analysis — {selected_product}",
                    template=PLOTLY_TEMPLATE,
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run `python train.py` to generate prediction data.")

    else:
        # Show all products comparison if available
        st.info("📊 Run `python train.py` to generate model comparison data.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("""
    <div class="glass-card">
        <p class="hero-title">🔍 Feature Importance Analysis</p>
        <p class="hero-subtitle">Understand which features drive demand predictions in the XGBoost model</p>
    </div>
    """, unsafe_allow_html=True)

    fi = load_feature_importance(selected_product)

    if fi is not None:
        # Sort by importance descending
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(fi))
        with col2:
            st.metric("#1 Feature", fi.iloc[0]["feature"])
        with col3:
            top3_pct = fi.head(3)["importance"].sum() / fi["importance"].sum() * 100
            st.metric("Top 3 Share", f"{top3_pct:.1f}%")
        with col4:
            top10_pct = fi.head(10)["importance"].sum() / fi["importance"].sum() * 100
            st.metric("Top 10 Share", f"{top10_pct:.1f}%")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📊 All Features", "🏆 Top 10 Deep Dive", "🌐 Cross-Product Comparison"])

        with tab1:
            # Horizontal bar chart of all features
            fi_sorted = fi.sort_values("importance", ascending=True)  # ascending for horizontal bar
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fi_sorted["importance"],
                y=fi_sorted["feature"],
                orientation="h",
                marker=dict(
                    color=fi_sorted["importance"],
                    colorscale=[
                        [0, "rgba(99, 102, 241, 0.3)"],
                        [0.5, COLORS["primary"]],
                        [1, COLORS["accent"]],
                    ],
                    cornerradius=6,
                ),
                text=[f"{v:.4f}" for v in fi_sorted["importance"]],
                textposition="outside",
                textfont=dict(color="#a5b4fc", size=11),
            ))
            fig.update_layout(
                title=f"Feature Importance — {selected_product} (XGBoost)",
                xaxis_title="Importance Score",
                template=PLOTLY_TEMPLATE,
                height=max(400, len(fi) * 28),
                margin=dict(l=200),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance table
            st.markdown("### 📋 Feature Importance Table")
            fi_display = fi.copy()
            fi_display["rank"] = range(1, len(fi_display) + 1)
            fi_display["pct_of_total"] = (fi_display["importance"] / fi_display["importance"].sum() * 100).round(2)
            fi_display = fi_display[["rank", "feature", "importance", "pct_of_total"]]
            fi_display.columns = ["Rank", "Feature", "Importance", "% of Total"]
            st.dataframe(fi_display, use_container_width=True, hide_index=True)

        with tab2:
            # Top 10 features with detailed breakdown
            top10 = fi.head(10).copy()

            col_left, col_right = st.columns(2)

            with col_left:
                # Donut chart of top 10 vs rest
                top10_imp = top10["importance"].sum()
                rest_imp = fi.iloc[10:]["importance"].sum() if len(fi) > 10 else 0

                fig = go.Figure(data=[go.Pie(
                    labels=["Top 10 Features", "Other Features"],
                    values=[top10_imp, rest_imp],
                    hole=0.65,
                    marker=dict(colors=[COLORS["primary"], "rgba(99, 102, 241, 0.15)"]),
                    textinfo="label+percent",
                    textfont=dict(color="white"),
                )])
                fig.update_layout(
                    title="Top 10 vs Other Features",
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    showlegend=False,
                    annotations=[{
                        "text": f"{top10_imp/(top10_imp+rest_imp)*100:.0f}%",
                        "x": 0.5, "y": 0.5,
                        "font_size": 28, "font_color": COLORS["primary"],
                        "showarrow": False,
                    }],
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Categorize features by type
                def categorize_feature(name):
                    if "lag" in name:
                        return "Lag Features"
                    elif "rolling" in name or "expanding" in name:
                        return "Rolling Stats"
                    elif name in ["month", "day_of_week", "year", "quarter", "day_of_month",
                                  "week_of_year", "is_weekend", "is_month_start", "is_month_end", "is_holiday"]:
                        return "Temporal"
                    elif name in ["price", "promotion"]:
                        return "Business"
                    else:
                        return "Other"

                fi_cats = fi.copy()
                fi_cats["category"] = fi_cats["feature"].apply(categorize_feature)
                cat_imp = fi_cats.groupby("category")["importance"].sum().reset_index()
                cat_imp = cat_imp.sort_values("importance", ascending=False)

                fig = go.Figure(data=[go.Pie(
                    labels=cat_imp["category"],
                    values=cat_imp["importance"],
                    hole=0.5,
                    marker=dict(colors=[
                        COLORS["primary"], COLORS["success"],
                        COLORS["warning"], COLORS["accent"],
                        COLORS["info"],
                    ]),
                    textinfo="label+percent",
                    textfont=dict(color="white"),
                )])
                fig.update_layout(
                    title="Importance by Feature Category",
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top 10 bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top10["feature"],
                y=top10["importance"],
                marker=dict(
                    color=top10["importance"],
                    colorscale=[[0, COLORS["info"]], [0.5, COLORS["primary"]], [1, COLORS["accent"]]],
                    cornerradius=8,
                ),
                text=[f"{v:.4f}" for v in top10["importance"]],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig.update_layout(
                title=f"Top 10 Features — {selected_product}",
                yaxis_title="Importance Score",
                template=PLOTLY_TEMPLATE,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Cross-product feature importance heatmap
            st.markdown("### 🌐 Feature Importance Across All Products")

            all_fi = {}
            for p in config.PRODUCTS:
                p_fi = load_feature_importance(p)
                if p_fi is not None:
                    all_fi[p] = dict(zip(p_fi["feature"], p_fi["importance"]))

            if len(all_fi) >= 2:
                fi_matrix = pd.DataFrame(all_fi)
                # Normalize each column to 0-1 for better comparison
                fi_norm = fi_matrix.div(fi_matrix.max())

                fig = go.Figure(data=go.Heatmap(
                    z=fi_norm.values,
                    x=fi_norm.columns.tolist(),
                    y=fi_norm.index.tolist(),
                    colorscale=[
                        [0, "rgba(15, 12, 41, 0.9)"],
                        [0.3, "rgba(99, 102, 241, 0.4)"],
                        [0.6, COLORS["primary"]],
                        [1, COLORS["accent"]],
                    ],
                    text=fi_matrix.values.round(4),
                    texttemplate="%{text:.3f}",
                    textfont=dict(size=10, color="white"),
                    hovertemplate="Feature: %{y}<br>Product: %{x}<br>Importance: %{text}<extra></extra>",
                ))
                fig.update_layout(
                    title="Feature Importance Heatmap (Normalized)",
                    template=PLOTLY_TEMPLATE,
                    height=max(500, len(fi_norm) * 28),
                    margin=dict(l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top feature per product
                st.markdown("### 🏆 Top Feature Per Product")
                top_per_product = []
                for p in all_fi:
                    p_fi_df = pd.DataFrame(list(all_fi[p].items()), columns=["feature", "importance"])
                    top = p_fi_df.sort_values("importance", ascending=False).iloc[0]
                    top_per_product.append({"Product": p, "Top Feature": top["feature"], "Importance": f"{top['importance']:.4f}"})
                st.dataframe(pd.DataFrame(top_per_product), use_container_width=True, hide_index=True)
            else:
                st.info("Need at least 2 products with feature importance data for cross-product comparison.")

    else:
        st.info("🔍 Run `python train.py` to generate feature importance data.")
