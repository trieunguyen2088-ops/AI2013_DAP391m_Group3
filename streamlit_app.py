from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"

st.set_page_config(
    page_title="M5 Inventory Replenishment Dashboard",
    page_icon="📦",
    layout="wide",
)

SCENARIO_LABELS = {
    "short_lead_time": "Short lead time",
    "base_case": "Base case",
    "long_lead_time": "Long lead time",
}

PREDICTION_COLUMNS = {
    "Seasonal Naive 28": "seasonal_naive_28",
    "Moving Average 28": "moving_average_28",
    "Random Forest": "random_forest_pred",
    "XGBoost": "xgboost_pred",
    "CatBoost": "catboost_pred",
    "LightGBM": "lightgbm_pred",
    "LightGBM Tuned": "lightgbm_tuned_pred",
}

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

@st.cache_data
def load_all_data():
    data = {
        "test_metrics": load_csv("test_metrics.csv"),
        "validation_metrics": load_csv("validation_metrics.csv"),
        "policy": load_csv("policy_comparison.csv"),
        "feature_importance": load_csv("feature_importance.csv"),
        "hyperparams": load_csv("lightgbm_hyperparameter_comparison.csv"),
        "forecast_ts": load_csv("forecast_timeseries_sample.csv"),
        "inventory_ts": load_csv("inventory_timeseries_sample.csv"),
    }
    data["forecast_ts"]["date"] = pd.to_datetime(data["forecast_ts"]["date"])
    data["inventory_ts"]["date"] = pd.to_datetime(data["inventory_ts"]["date"])
    return data

def format_currency(value):
    return f"{value:,.2f}"

def clean_scenario(df):
    out = df.copy()
    out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS).fillna(out["scenario"])
    return out

def show_kpis(policy_df, scenario):
    selected = policy_df[policy_df["scenario"] == scenario]
    best_cost_row = selected.loc[selected["total_cost"].idxmin()]
    best_service_row = selected.loc[selected["service_level_achieved"].idxmax()]
    tuned = selected[selected["forecast_model"] == "LightGBM Tuned"]
    lightgbm = selected[selected["forecast_model"] == "LightGBM"]
    fixed = selected[selected["forecast_model"] == "Fixed Rule"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest total cost", best_cost_row["forecast_model"], format_currency(best_cost_row["total_cost"]))
    c2.metric("Highest service level", best_service_row["forecast_model"], f"{best_service_row['service_level_achieved']:.2%}")
    if not tuned.empty:
        c3.metric("Tuned LightGBM cost", format_currency(float(tuned["total_cost"].iloc[0])))
    if not fixed.empty and not tuned.empty:
        saving = float(tuned["relative_cost_savings_pct"].iloc[0])
        c4.metric("Tuned vs fixed rule", f"{saving:.2f}%")

def overview_page(data):
    st.title("📦 M5 Forecast-Driven Inventory Replenishment Dashboard")
    st.markdown(
        """
        This Streamlit app demonstrates how forecast outputs can be converted into inventory replenishment decisions. It uses the Walmart M5 retail sales setting and compares forecasting performance with downstream inventory simulation cost.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.info("**Dataset**\n\nWalmart M5 sales signals")
    c2.info("**Forecasting models**\n\nNaive, Random Forest, XGBoost, CatBoost, LightGBM, Tuned LightGBM")
    c3.info("**Inventory policy**\n\nForecast-driven reorder point simulation")

    st.subheader("Research workflow")
    st.markdown(
        """
        **Sales data** → **Feature engineering** → **Forecasting models** → **Reorder point simulation** → **Cost comparison**
        """
    )

    fig_path = ASSET_DIR / "fig_01_chronological_split.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Chronological train-validation-test split used in the project")

    st.subheader("Main dashboard outputs")
    st.write(
        "The app focuses on small processed output files, not the full raw M5 data. This keeps deployment fast and avoids GitHub file-size problems."
    )

    with open(DATA_DIR / "app_data_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    st.json(summary, expanded=False)

def forecasting_page(data):
    st.title("📈 Forecasting Performance")
    split = st.radio("Select evaluation split", ["Test", "Validation"], horizontal=True)
    metrics = data["test_metrics"] if split == "Test" else data["validation_metrics"]

    st.subheader(f"{split} metrics")
    st.dataframe(metrics, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics, x="model", y="RMSSE", title="RMSSE by model", text_auto=".3f")
        fig.update_layout(xaxis_title="Model", yaxis_title="RMSSE")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(metrics, x="model", y="weighted_RMSSE_bottom", title="Weighted RMSSE by model", text_auto=".3f")
        fig.update_layout(xaxis_title="Model", yaxis_title="Weighted RMSSE")
        st.plotly_chart(fig, use_container_width=True)

    best = metrics.sort_values("RMSSE").iloc[0]
    st.success(f"Best RMSSE on the {split.lower()} split: {best['model']} ({best['RMSSE']:.4f}).")

    st.subheader("LightGBM tuning explanation")
    st.dataframe(data["hyperparams"], use_container_width=True, hide_index=True)

    st.subheader("Feature importance")
    imp = data["feature_importance"].copy()
    model_choices = sorted(imp["model"].unique())
    model = st.selectbox("Select model", model_choices)
    top_n = st.slider("Number of features", min_value=5, max_value=30, value=15)
    imp_plot = imp[imp["model"] == model].sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(imp_plot.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top {top_n} features: {model}")
    st.plotly_chart(fig, use_container_width=True)

def forecast_visual_page(data):
    st.title("🔍 Actual vs Forecast Visualization")
    df = data["forecast_ts"].copy()

    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Split", sorted(df["split"].unique()))
    item = c2.selectbox("Item/store series", sorted(df[df["split"] == split]["id"].unique()))
    available_models = [m for m, col in PREDICTION_COLUMNS.items() if col in df.columns]
    model = c3.selectbox("Forecast model", available_models, index=available_models.index("LightGBM Tuned") if "LightGBM Tuned" in available_models else 0)

    pred_col = PREDICTION_COLUMNS[model]
    plot_df = df[(df["split"] == split) & (df["id"] == item)][["date", "sales", pred_col]].copy()
    plot_df = plot_df.rename(columns={"sales": "Actual demand", pred_col: model})
    long = plot_df.melt("date", var_name="Series", value_name="Demand")

    fig = px.line(long, x="date", y="Demand", color="Series", markers=True, title=f"Actual vs forecast: {item}")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(plot_df, use_container_width=True, hide_index=True)

def inventory_results_page(data):
    st.title("🏬 Inventory Simulation Results")
    policy = clean_scenario(data["policy"])
    scenario_labels = list(SCENARIO_LABELS.values())
    label_to_key = {v: k for k, v in SCENARIO_LABELS.items()}
    selected_label = st.selectbox("Scenario", scenario_labels, index=scenario_labels.index("Base case"))
    scenario = label_to_key[selected_label]

    show_kpis(policy, scenario)

    filtered = policy[policy["scenario"] == scenario].sort_values("total_cost")
    st.subheader("Policy comparison table")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(filtered, x="forecast_model", y="total_cost", title="Total cost by model", text_auto=".2s")
        fig.update_layout(xaxis_title="Forecast model", yaxis_title="Total cost")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cost_cols = ["holding_cost", "ordering_cost", "stockout_cost"]
        stacked = filtered[["forecast_model"] + cost_cols].melt("forecast_model", var_name="Cost type", value_name="Cost")
        fig = px.bar(stacked, x="forecast_model", y="Cost", color="Cost type", title="Cost components by model")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost across lead-time scenarios")
    fig = px.bar(policy, x="scenario_label", y="total_cost", color="forecast_model", barmode="group", title="Total cost across scenarios")
    fig.update_layout(xaxis_title="Scenario", yaxis_title="Total cost")
    st.plotly_chart(fig, use_container_width=True)

def inventory_timeseries_page(data):
    st.title("📉 Inventory Time-series Explorer")
    df = data["inventory_ts"].copy()
    scenario_labels = {k: SCENARIO_LABELS.get(k, k) for k in df["scenario"].unique()}
    label_to_key = {v: k for k, v in scenario_labels.items()}

    c1, c2, c3 = st.columns(3)
    scenario_label = c1.selectbox("Scenario", sorted(label_to_key.keys()))
    scenario = label_to_key[scenario_label]
    model_options = sorted(df[df["scenario"] == scenario]["forecast_model"].dropna().unique())
    model = c2.selectbox("Forecast model / policy", model_options)
    item_options = sorted(df[(df["scenario"] == scenario) & (df["forecast_model"] == model)]["id"].unique())
    item = c3.selectbox("Item/store series", item_options)

    plot_df = df[(df["scenario"] == scenario) & (df["forecast_model"] == model) & (df["id"] == item)].sort_values("date")

    st.subheader("Inventory level, reorder point, and orders")
    line_cols = [c for c in ["on_hand_inventory", "reorder_point", "inventory_position"] if c in plot_df.columns]
    long = plot_df[["date"] + line_cols].melt("date", var_name="Series", value_name="Value")
    fig = px.line(long, x="date", y="Value", color="Series", markers=True, title=f"Inventory simulation: {item}")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        demand_cols = [c for c in ["actual_demand", "selected_forecast"] if c in plot_df.columns]
        dlong = plot_df[["date"] + demand_cols].melt("date", var_name="Series", value_name="Demand")
        fig = px.line(dlong, x="date", y="Demand", color="Series", markers=True, title="Demand and selected forecast")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(plot_df, x="date", y="order_quantity", title="Order quantity over time")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(plot_df, use_container_width=True, hide_index=True)

def what_if_page(data):
    st.title("🧪 What-if Replenishment Calculator")
    st.markdown(
        "This lightweight calculator illustrates the reorder point logic used by forecast-driven inventory policies. It is not retraining the model; it lets you test simple replenishment assumptions."
    )

    c1, c2, c3 = st.columns(3)
    avg_daily_demand = c1.number_input("Average daily demand", min_value=0.0, value=5.0, step=0.5)
    demand_std = c2.number_input("Demand standard deviation", min_value=0.0, value=2.0, step=0.5)
    lead_time = c3.slider("Lead time in days", min_value=1, max_value=30, value=7)

    c4, c5, c6 = st.columns(3)
    service_level = c4.selectbox("Service level", [0.90, 0.95, 0.975, 0.99], index=1)
    current_inventory_position = c5.number_input("Current inventory position", min_value=0.0, value=20.0, step=1.0)
    review_period = c6.slider("Review period", min_value=1, max_value=14, value=7)

    z_lookup = {0.90: 1.28, 0.95: 1.65, 0.975: 1.96, 0.99: 2.33}
    z = z_lookup[service_level]
    lead_time_demand = avg_daily_demand * lead_time
    safety_stock = z * demand_std * np.sqrt(lead_time)
    reorder_point = lead_time_demand + safety_stock
    target_inventory = reorder_point + avg_daily_demand * review_period
    order_quantity = max(0, target_inventory - current_inventory_position)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lead-time demand", f"{lead_time_demand:.2f}")
    c2.metric("Safety stock", f"{safety_stock:.2f}")
    c3.metric("Reorder point", f"{reorder_point:.2f}")
    c4.metric("Suggested order quantity", f"{order_quantity:.2f}")

    explanation = pd.DataFrame(
        {
            "Component": ["Lead-time demand", "Safety stock", "Reorder point", "Target inventory", "Order quantity"],
            "Value": [lead_time_demand, safety_stock, reorder_point, target_inventory, order_quantity],
        }
    )
    st.dataframe(explanation, use_container_width=True, hide_index=True)

def conclusion_page(data):
    st.title("✅ Final Comparison and Research Takeaways")
    metrics = data["test_metrics"].copy()
    policy = clean_scenario(data["policy"])

    st.subheader("Forecasting vs inventory trade-off")
    pivot = policy.pivot_table(index="forecast_model", columns="scenario_label", values="total_cost", aggfunc="min").reset_index()
    merged = metrics.rename(columns={"model": "forecast_model"}).merge(pivot, on="forecast_model", how="left")
    st.dataframe(merged, use_container_width=True, hide_index=True)

    best_forecast = metrics.sort_values("RMSSE").iloc[0]
    best_cost = policy.sort_values("total_cost").iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.success(f"Best forecasting model\n\n**{best_forecast['model']}**")
    c2.success(f"Lowest single-scenario cost\n\n**{best_cost['forecast_model']}**")
    c3.info("Practical interpretation\n\n**Tuned LightGBM is a strong accuracy-cost compromise.**")

    st.markdown(
        """
        **Main takeaway:** forecasting accuracy and inventory cost are related, but they are not identical objectives. The tuned LightGBM model achieves the strongest RMSSE-based forecasting result in this experiment. In the inventory simulation, standard LightGBM can produce lower total cost in some scenarios, while tuned LightGBM still remains competitive and improves substantially over simpler replenishment baselines.
        """
    )

    fig_path = ASSET_DIR / "fig3.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Project visualization output")


def main():
    data = load_all_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Forecasting Performance",
            "Actual vs Forecast",
            "Inventory Simulation Results",
            "Inventory Time-series Explorer",
            "What-if Simulator",
            "Final Comparison",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("AI2013 / DAP391m Group 3")

    if page == "Overview":
        overview_page(data)
    elif page == "Forecasting Performance":
        forecasting_page(data)
    elif page == "Actual vs Forecast":
        forecast_visual_page(data)
    elif page == "Inventory Simulation Results":
        inventory_results_page(data)
    elif page == "Inventory Time-series Explorer":
        inventory_timeseries_page(data)
    elif page == "What-if Simulator":
        what_if_page(data)
    elif page == "Final Comparison":
        conclusion_page(data)

if __name__ == "__main__":
    main()
