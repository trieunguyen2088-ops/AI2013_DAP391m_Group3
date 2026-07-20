from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"

MODEL_COLUMNS = {
    "Seasonal Naive 28": "seasonal_naive_28",
    "Moving Average 28": "moving_average_28",
    "XGBoost": "xgboost_pred",
    "LightGBM": "lightgbm_pred",
    "LightGBM Tuned": "lightgbm_tuned_pred",
}

SAFETY_COLUMNS = {
    "Seasonal Naive 28": "ss_seasonal_naive_28",
    "Moving Average 28": "ss_moving_average_28",
    "XGBoost": "ss_xgboost_pred",
    "LightGBM": "ss_lightgbm_pred",
    "LightGBM Tuned": "ss_lightgbm_tuned_pred",
}

MODEL_ARTIFACTS = {
    "LightGBM": PROJECT_DIR / "models" / "lightgbm_final_model.pkl",
    "LightGBM Tuned": PROJECT_DIR / "models" / "lightgbm_tuned_final_model.pkl",
    "XGBoost": PROJECT_DIR / "models" / "xgboost_final_model.pkl",
}


st.set_page_config(
    page_title="M5 Segment-Aware Replenishment",
    page_icon="DAP",
    layout="wide",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
        div[data-testid="stMetric"] {
            background: rgba(240, 245, 250, 0.78);
            border: 1px solid rgba(120, 140, 165, 0.25);
            padding: 0.75rem 0.85rem;
            border-radius: 8px;
        }
        .dap-note {
            border-left: 4px solid #173A67;
            background: rgba(220, 235, 250, 0.55);
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin: 0.5rem 0 1rem 0;
        }
        .dap-warning {
            border-left: 4px solid #B7791F;
            background: rgba(255, 241, 200, 0.70);
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin: 0.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(name: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=parse_dates)


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    with open(DATA_DIR / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=True)
def load_data() -> Dict[str, pd.DataFrame | dict]:
    return {
        "summary": read_json("final_experiment_summary.json"),
        "forecast_metrics": read_csv("forecast_metrics_main_models.csv"),
        "test_metrics_weighted": read_csv("test_metrics_weighted.csv"),
        "validation_metrics_weighted": read_csv("validation_metrics_weighted.csv"),
        "tuning": read_csv("lightgbm_manual_tuning_results.csv"),
        "feature_importance": read_csv("feature_importance.csv"),
        "policy": read_csv("final_policy_comparison.csv"),
        "group_policy": read_csv("final_group_policy_comparison.csv"),
        "segment_params": read_csv("selected_segment_policy_parameters.csv"),
        "stats": read_csv("global_vs_segment_policy_statistics.csv"),
        "group_counts": read_csv("demand_group_counts.csv"),
        "profiles": read_csv("validation_inventory_profiles.csv"),
        "forecast_ts": read_csv("forecast_timeseries.csv", parse_dates=["date"]),
        "inventory_sample": read_csv("inventory_daily_sample.csv", parse_dates=["date"]),
        "model_artifacts": read_json("model_artifacts.json"),
    }


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def model_status() -> pd.DataFrame:
    rows = []
    for label, file_path in MODEL_ARTIFACTS.items():
        rows.append(
            {
                "model": label,
                "artifact_path": str(file_path),
                "exists": file_path.exists(),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2) if file_path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_model(model_label: str):
    import joblib

    file_path = MODEL_ARTIFACTS[model_label]
    return joblib.load(file_path)


@st.cache_data(show_spinner=False)
def load_feature_frame() -> Tuple[pd.DataFrame, list[str]]:
    features_path = PROJECT_DIR / "data" / "processed" / "test_features.parquet"
    metadata_path = PROJECT_DIR / "data" / "processed" / "feature_metadata.json"
    df = pd.read_parquet(features_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return df, metadata["feature_columns"]


def live_model_prediction(model_label: str, series_id: str, d_num: int) -> Optional[float]:
    if model_label not in MODEL_ARTIFACTS or not MODEL_ARTIFACTS[model_label].exists():
        return None
    try:
        feature_df, features = load_feature_frame()
        row = feature_df[(feature_df["id"] == series_id) & (feature_df["d_num"] == d_num)]
        if row.empty:
            return None
        model = load_model(model_label)
        pred = model.predict(row[features])[0]
        return max(0.0, float(pred))
    except Exception as exc:
        st.warning(f"Saved model inference could not run: {exc}")
        return None


def page_overview(data: Dict[str, pd.DataFrame | dict]) -> None:
    summary = data["summary"]
    policy = data["policy"].copy()
    forecast = data["forecast_metrics"].copy()
    best_policy = policy.sort_values("total_cost").iloc[0]
    best_forecast = forecast.sort_values("RMSSE").iloc[0]

    st.title("Forecast-Based Segment-Aware Replenishment Simulation")
    st.markdown(
        """
        <div class="dap-note">
        The dashboard summarizes the final notebook outputs. Forecasts are converted into reorder-point decisions,
        then compared using simulated total cost and fill rate. Inventory values are controlled simulation outputs,
        not actual Walmart inventory measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Item-store series", f"{summary['n_series']:,}")
    c2.metric("Forecast models", len(summary["forecast_models"]))
    c3.metric("Best forecast RMSSE", best_forecast["model"], f"{best_forecast['RMSSE']:.4f}")
    c4.metric("Lowest cost policy", best_policy["policy_label"], fmt_num(best_policy["total_cost"]))

    image_path = ASSET_DIR / "fig_00_forecast_to_replenishment_pipeline.png"
    if image_path.exists():
        st.image(str(image_path), caption="Forecast-to-replenishment workflow")

    st.subheader("Project scope")
    st.write(
        "The app uses five forecast models: Seasonal Naive 28, Moving Average 28, XGBoost, LightGBM, and LightGBM Tuned. "
        "The inventory layer compares seven policies, including Global LightGBM ROP, LightGBM Tuned ROP, and Segment-Aware LightGBM ROP."
    )

    st.subheader("Saved model artifacts")
    st.dataframe(model_status(), use_container_width=True, hide_index=True)


def page_forecasting(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Forecasting Performance")
    metrics = data["forecast_metrics"].copy()
    weighted = data["test_metrics_weighted"].copy()
    tuning = data["tuning"].copy()
    feature_importance = data["feature_importance"].copy()

    best = metrics.sort_values("RMSSE").iloc[0]
    st.success(
        f"Best test RMSSE: {best['model']} with RMSSE = {best['RMSSE']:.4f}. "
        "Standard LightGBM still has a slightly lower RMSE than the tuned model."
    )

    st.subheader("Main forecast metrics used in the paper")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics.sort_values("RMSSE"), x="model", y="RMSSE", text_auto=".4f", title="Test RMSSE")
        fig.update_layout(xaxis_title="", yaxis_title="RMSSE")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(weighted.sort_values("weighted_RMSSE_bottom"), x="model", y="weighted_RMSSE_bottom", text_auto=".4f", title="Weighted RMSSE")
        fig.update_layout(xaxis_title="", yaxis_title="Weighted RMSSE")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Manual LightGBM tuning candidates")
    st.caption("The tuned model is selected from a small validation search. No Optuna search is used.")
    st.dataframe(tuning, use_container_width=True, hide_index=True)

    st.subheader("Feature importance")
    available_models = sorted(feature_importance["model"].dropna().unique())
    selected_model = st.selectbox("Model", available_models, index=available_models.index("LightGBM Tuned") if "LightGBM Tuned" in available_models else 0)
    top_n = st.slider("Number of features", 5, 25, 15)
    imp = feature_importance[feature_importance["model"] == selected_model].sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top {top_n} features: {selected_model}")
    st.plotly_chart(fig, use_container_width=True)


def page_inventory(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Inventory Simulation Results")
    policy = data["policy"].copy().sort_values("total_cost")
    best = policy.iloc[0]

    st.markdown(
        """
        <div class="dap-warning">
        Cost, lead time, and stockout penalties are simulation assumptions. The comparison measures policy behavior
        under the same controlled setting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest total cost", best["policy_label"], fmt_num(best["total_cost"]))
    c2.metric("Fill rate", fmt_pct(best["fill_rate"]))
    c3.metric("Saving vs fixed", f"{best['relative_cost_savings_vs_fixed_pct']:.2f}%")
    c4.metric("Total orders", f"{int(best['total_orders']):,}")

    st.subheader("Policy comparison")
    st.dataframe(policy, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(policy, x="policy_label", y="total_cost", text_auto=".2s", title="Total simulated cost")
        fig.update_layout(xaxis_title="", yaxis_title="Total cost")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cost_parts = policy[["policy_label", "holding_cost", "ordering_cost", "stockout_cost"]].melt(
            "policy_label", var_name="cost_component", value_name="cost"
        )
        fig = px.bar(cost_parts, x="policy_label", y="cost", color="cost_component", title="Cost components")
        fig.update_layout(xaxis_title="", yaxis_title="Cost")
        st.plotly_chart(fig, use_container_width=True)

    fig_path = ASSET_DIR / "fig_07_policy_cost_fill_tradeoff.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Cost and fill-rate trade-off")


def page_segments(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Demand Segments and Segment-Aware Calibration")
    counts = data["group_counts"].copy()
    params = data["segment_params"].copy()
    group_policy = data["group_policy"].copy()
    stats = data["stats"].copy()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Training-only group counts")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        fig = px.pie(counts, names="demand_group", values="series_count", title="Series distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Selected segment parameters")
        st.dataframe(params, use_container_width=True, hide_index=True)
        fig = px.scatter(
            params,
            x="coverage_days",
            y="safety_multiplier",
            size="validation_total_cost",
            color="demand_group",
            hover_data=["validation_fill_rate", "validation_lost_sales"],
            title="Validation-selected coverage and safety settings",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global versus segment-aware LightGBM")
    compare = group_policy[group_policy["policy_label"].isin(["Global LightGBM ROP", "Segment-Aware LightGBM ROP"])]
    st.dataframe(compare, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(compare, x="demand_group", y="cost_per_demand_unit", color="policy_label", barmode="group", title="Cost per demand unit")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(compare, x="demand_group", y="fill_rate", color="policy_label", markers=True, title="Fill rate by group")
        fig.add_hline(y=0.95, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    row = stats.iloc[0]
    st.info(
        f"Paired item-store comparison: mean cost difference = {row['mean_cost_difference']:.4f}, "
        f"median = {row['median_cost_difference']:.4f}, improved series = {row['share_series_improved']:.2%}, "
        f"Wilcoxon p-value = {row['wilcoxon_p_value']:.6e}."
    )


def sum_forecast_window(values: np.ndarray, start_idx: int, days: int) -> float:
    window = values[start_idx : start_idx + days]
    if len(window) >= days:
        return float(np.clip(window, 0, None).sum())
    if len(window) == 0:
        return 0.0
    pad_value = float(np.clip(window, 0, None).mean())
    return float(np.clip(window, 0, None).sum() + pad_value * (days - len(window)))


def get_recommendation(
    series_df: pd.DataFrame,
    profile: pd.Series,
    model_label: str,
    selected_date: pd.Timestamp,
    lead_time: int,
    coverage_days: int,
    safety_multiplier: float,
    on_hand: float,
    incoming: float,
) -> dict:
    pred_col = MODEL_COLUMNS[model_label]
    ss_col = SAFETY_COLUMNS[model_label]
    ordered = series_df.sort_values("date").reset_index(drop=True)
    start_idx = int(ordered.index[ordered["date"] == selected_date][0])
    values = ordered[pred_col].astype(float).to_numpy()
    lead_forecast = sum_forecast_window(values, start_idx, lead_time)
    target_forecast = sum_forecast_window(values, start_idx, lead_time + coverage_days)
    safety_stock = max(0.0, float(profile.get(ss_col, 0.0))) * safety_multiplier
    reorder_point = lead_forecast + safety_stock
    target_inventory = target_forecast + safety_stock
    inventory_position = on_hand + incoming
    should_order = inventory_position <= reorder_point
    suggested_order = max(0, math.ceil(target_inventory - inventory_position)) if should_order else 0
    return {
        "lead_forecast": lead_forecast,
        "target_forecast": target_forecast,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_inventory": target_inventory,
        "inventory_position": inventory_position,
        "suggested_order": suggested_order,
        "should_order": should_order,
    }


def page_recommendation(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Order Recommendation")
    st.markdown(
        """
        <div class="dap-note">
        The calculator uses saved forecasts and validation safety-stock profiles. It answers: given current stock and incoming orders,
        how many units should be ordered under the selected policy assumptions?
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecast = data["forecast_ts"].copy()
    profiles = data["profiles"].copy()
    params = data["segment_params"].copy().set_index("demand_group")

    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Forecast split", ["Test", "Validation"], index=0)
    split_df = forecast[forecast["split"] == split].copy()
    state = c2.selectbox("State", sorted(split_df["state_id"].dropna().unique()))
    store = c3.selectbox("Store", sorted(split_df[split_df["state_id"] == state]["store_id"].dropna().unique()))

    store_df = split_df[split_df["store_id"] == store].copy()
    labels = (
        store_df[["id", "item_id", "dept_id", "cat_id"]]
        .drop_duplicates()
        .assign(label=lambda d: d["item_id"] + " | " + d["id"])
        .sort_values("label")
    )
    selected_label = st.selectbox("Item-store series", labels["label"].tolist())
    series_id = labels.loc[labels["label"] == selected_label, "id"].iloc[0]
    series_df = store_df[store_df["id"] == series_id].sort_values("date").copy()
    profile_match = profiles[profiles["id"] == series_id]
    if profile_match.empty:
        st.error("No validation profile found for the selected series.")
        return
    profile = profile_match.iloc[0]
    demand_group = str(profile["demand_group"])

    c1, c2, c3, c4 = st.columns(4)
    selected_date = c1.selectbox("Decision date", series_df["date"].dt.date.tolist())
    selected_date = pd.Timestamp(selected_date)
    policy_mode = c2.selectbox(
        "Policy mode",
        ["Segment-Aware LightGBM ROP", "Global forecast ROP"],
        index=0,
    )
    if policy_mode == "Segment-Aware LightGBM ROP":
        model_label = "LightGBM"
        selected_param = params.loc[demand_group]
        coverage_days = int(selected_param["coverage_days"])
        safety_multiplier = float(selected_param["safety_multiplier"])
        c3.metric("Demand group", demand_group)
        c4.metric("Selected params", f"C={coverage_days}, k={safety_multiplier:.2f}")
    else:
        model_label = c3.selectbox("Forecast model", list(MODEL_COLUMNS.keys()), index=3)
        coverage_days = int(c4.number_input("Coverage days", min_value=1, max_value=28, value=7, step=1))
        safety_multiplier = float(st.slider("Safety multiplier", min_value=0.0, max_value=2.5, value=1.0, step=0.05))

    c1, c2, c3 = st.columns(3)
    lead_time = int(c1.number_input("Lead time days", min_value=1, max_value=28, value=7, step=1))
    on_hand = float(c2.number_input("Current on-hand inventory", min_value=0.0, value=10.0, step=1.0))
    incoming = float(c3.number_input("Incoming open orders", min_value=0.0, value=0.0, step=1.0))

    rec = get_recommendation(
        series_df=series_df,
        profile=profile,
        model_label=model_label,
        selected_date=selected_date,
        lead_time=lead_time,
        coverage_days=coverage_days,
        safety_multiplier=safety_multiplier,
        on_hand=on_hand,
        incoming=incoming,
    )

    st.subheader("Recommendation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory position", fmt_num(rec["inventory_position"]))
    c2.metric("Reorder point", fmt_num(rec["reorder_point"]))
    c3.metric("Target inventory", fmt_num(rec["target_inventory"]))
    c4.metric("Suggested order", f"{int(rec['suggested_order']):,} units")

    if rec["should_order"]:
        st.success("Decision: place a replenishment order.")
    else:
        st.info("Decision: no order is needed because inventory position is above the reorder point.")

    detail = pd.DataFrame(
        [
            ["Forecasted lead-time demand", rec["lead_forecast"]],
            ["Forecasted lead-time plus coverage demand", rec["target_forecast"]],
            ["Safety stock after multiplier", rec["safety_stock"]],
            ["Inventory position", rec["inventory_position"]],
            ["Reorder point", rec["reorder_point"]],
            ["Target inventory", rec["target_inventory"]],
            ["Suggested order quantity", rec["suggested_order"]],
        ],
        columns=["Component", "Value"],
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    plot_df = series_df[["date", "sales", MODEL_COLUMNS[model_label]]].rename(
        columns={"sales": "Actual demand", MODEL_COLUMNS[model_label]: f"{model_label} forecast"}
    )
    long = plot_df.melt("date", var_name="Series", value_name="Demand")
    fig = px.line(long, x="date", y="Demand", color="Series", markers=True, title=f"Actual demand and forecast: {series_id}")
    fig.add_vline(x=selected_date, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    if model_label in MODEL_ARTIFACTS:
        with st.expander("Optional saved model inference check"):
            st.write("For LightGBM, LightGBM Tuned, and XGBoost, the app can load the saved model artifact and predict the selected test feature row when available.")
            if st.button("Run saved model inference for selected date"):
                row = series_df[series_df["date"] == selected_date].iloc[0]
                live_pred = live_model_prediction(model_label, series_id, int(row["d_num"]))
                if live_pred is None:
                    st.warning("No live prediction was produced. The app continues using saved forecast outputs.")
                else:
                    saved_pred = float(row[MODEL_COLUMNS[model_label]])
                    st.write(
                        pd.DataFrame(
                            [
                                {"source": "saved forecast CSV", "prediction": saved_pred},
                                {"source": "model artifact inference", "prediction": live_pred},
                            ]
                        )
                    )


def page_timeline(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Inventory Timeline Explorer")
    inv = data["inventory_sample"].copy()
    if inv.empty:
        st.warning("No inventory sample is available.")
        return

    c1, c2, c3 = st.columns(3)
    policy = c1.selectbox("Policy", sorted(inv["policy_label"].dropna().unique()))
    policy_df = inv[inv["policy_label"] == policy]
    group = c2.selectbox("Demand group", sorted(policy_df["demand_group"].dropna().unique()))
    group_df = policy_df[policy_df["demand_group"] == group]
    series_id = c3.selectbox("Series", sorted(group_df["id"].dropna().unique()))
    plot_df = group_df[group_df["id"] == series_id].sort_values("date")

    line_cols = ["on_hand_inventory", "inventory_position", "reorder_point", "target_inventory"]
    long = plot_df[["date"] + line_cols].melt("date", var_name="Series", value_name="Value")
    fig = px.line(long, x="date", y="Value", color="Series", markers=True, title=f"Inventory path: {series_id}")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(plot_df, x="date", y="order_quantity", title="Order quantity")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        demand_cols = ["actual_demand", "selected_forecast"]
        available = [col for col in demand_cols if col in plot_df.columns]
        long_demand = plot_df[["date"] + available].melt("date", var_name="Series", value_name="Demand")
        fig = px.line(long_demand, x="date", y="Demand", color="Series", markers=True, title="Demand and selected forecast")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(plot_df, use_container_width=True, hide_index=True)


def make_context_text(data: Dict[str, pd.DataFrame | dict]) -> str:
    forecast = data["forecast_metrics"].sort_values("RMSSE").iloc[0]
    policy = data["policy"].sort_values("total_cost").iloc[0]
    stats = data["stats"].iloc[0]
    return (
        f"Best forecast by RMSSE: {forecast['model']} ({forecast['RMSSE']:.4f}). "
        f"Lowest cost policy: {policy['policy_label']} with total cost {policy['total_cost']:.2f} and fill rate {policy['fill_rate']:.4f}. "
        f"Wilcoxon p-value for global LightGBM minus segment-aware LightGBM: {stats['wilcoxon_p_value']:.6e}. "
        "Inventory outputs are controlled simulation results, not actual Walmart operations."
    )


def rule_based_answer(question: str, data: Dict[str, pd.DataFrame | dict]) -> str:
    q = question.lower()
    forecast = data["forecast_metrics"].copy()
    policy = data["policy"].copy()
    stats = data["stats"].iloc[0]
    best_forecast = forecast.sort_values("RMSSE").iloc[0]
    best_policy = policy.sort_values("total_cost").iloc[0]

    if any(key in q for key in ["best", "tốt nhất", "tot nhat", "forecast", "rmsse", "model"]):
        return (
            f"Best forecast by RMSSE is {best_forecast['model']} with RMSSE = {best_forecast['RMSSE']:.4f}. "
            "LightGBM Tuned also has the lowest MAE, but standard LightGBM has a slightly lower RMSE."
        )
    if any(key in q for key in ["cost", "inventory", "policy", "rop", "replenishment", "chi phi", "ton kho"]):
        return (
            f"Lowest simulated total cost is {best_policy['policy_label']} at {best_policy['total_cost']:,.2f}, "
            f"with fill rate {best_policy['fill_rate']:.2%}. The saving versus fixed rule is "
            f"{best_policy['relative_cost_savings_vs_fixed_pct']:.2f}%."
        )
    if any(key in q for key in ["tuned", "lightgbm tuned"]):
        tuned_forecast = forecast[forecast["model"] == "LightGBM Tuned"].iloc[0]
        tuned_policy = policy[policy["policy_label"] == "LightGBM Tuned ROP"].iloc[0]
        return (
            f"LightGBM Tuned improves forecast MAE and RMSSE. In inventory simulation, LightGBM Tuned ROP costs "
            f"{tuned_policy['total_cost']:,.2f}, slightly above Global LightGBM ROP, so the paper should describe a trade-off honestly."
        )
    if any(key in q for key in ["simulation", "walmart", "actual", "real", "m5"]):
        return (
            "M5 provides sales, calendar, and price data but not real stock, purchase orders, lead time, or cost records. "
            "Inventory results in the app are controlled simulation outputs."
        )
    if any(key in q for key in ["wilcoxon", "statistical", "p-value", "p value"]):
        return (
            f"The paired Wilcoxon p-value is {stats['wilcoxon_p_value']:.6e}. "
            f"Mean cost difference is {stats['mean_cost_difference']:.4f}, and {stats['share_series_improved']:.2%} of series improve."
        )
    return make_context_text(data)


def gemini_answer(question: str, data: Dict[str, pd.DataFrame | dict]) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = (
            "Answer briefly using only the following project facts. "
            "Do not invent numbers. "
            f"Facts: {make_context_text(data)} "
            f"Question: {question}"
        )
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (response.text or "").strip() or None
    except Exception as exc:
        st.caption(f"Gemini fallback used because API call failed: {exc}")
        return None


def page_assistant(data: Dict[str, pd.DataFrame | dict]) -> None:
    st.title("Research Q&A Assistant")
    st.write("Ask about the data, models, inventory policies, LightGBM Tuned, or simulation assumptions.")
    question = st.text_input("Question", placeholder="Example: Why is Segment-Aware LightGBM better for cost?")
    if st.button("Ask") and question.strip():
        answer = gemini_answer(question, data) or rule_based_answer(question, data)
        st.markdown(answer)

    with st.expander("Context used by the assistant"):
        st.write(make_context_text(data))


def main() -> None:
    inject_style()
    data = load_data()
    pages = {
        "Overview": page_overview,
        "Forecasting": page_forecasting,
        "Inventory policies": page_inventory,
        "Segments": page_segments,
        "Order recommendation": page_recommendation,
        "Inventory timeline": page_timeline,
        "Research assistant": page_assistant,
    }
    st.sidebar.title("DAP391m Dashboard")
    st.sidebar.caption("Forecast-driven inventory replenishment")
    page_name = st.sidebar.radio("Page", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption("Numbers come from saved notebook outputs.")
    pages[page_name](data)


if __name__ == "__main__":
    main()
