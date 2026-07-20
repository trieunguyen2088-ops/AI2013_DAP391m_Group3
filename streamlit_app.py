from pathlib import Path
import json
import math
import os
from urllib.parse import quote
import html
import uuid
import time
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name == "streamlit_app" else APP_DIR
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
    "XGBoost": "xgboost_pred",
    "LightGBM": "lightgbm_pred",
    "LightGBM Tuned": "lightgbm_tuned_pred",
}

MAIN_FORECAST_MODELS = list(PREDICTION_COLUMNS.keys())
SAFETY_COLUMNS = {
    "Seasonal Naive 28": "ss_seasonal_naive_28",
    "Moving Average 28": "ss_moving_average_28",
    "XGBoost": "ss_xgboost_pred",
    "LightGBM": "ss_lightgbm_pred",
    "LightGBM Tuned": "ss_lightgbm_tuned_pred",
}
POLICY_MODEL_MAP = {
    "Seasonal Naive 28": "Seasonal Naive 28",
    "Seasonal Naive 28 ROP": "Seasonal Naive 28",
    "Moving Average 28": "Moving Average 28",
    "XGBoost ROP": "XGBoost",
    "Global LightGBM ROP": "LightGBM",
    "LightGBM Tuned ROP": "LightGBM Tuned",
    "Segment-Aware LightGBM ROP": "LightGBM",
    "Fixed rule": None,
}
EXCLUDED_ALL_MODELS = set()
EXCLUDED_SIMULATION_MODELS = set()
EXCLUDED_FEATURES = {"snap"}


NAV_ITEMS = [
    ("Overview", "🏠", "Project overview"),
    ("Forecasting Performance", "📈", "Forecast metrics"),
    ("Actual vs Forecast", "🔍", "Demand visualization"),
    ("Inventory Simulation Results", "🏬", "Cost simulation"),
    ("Segment Analysis", "🧩", "Demand groups"),
    ("Inventory Time-series Explorer", "📉", "Inventory timeline"),
    ("Order Recommendation", "🧪", "Reorder calculator"),
    ("Final Comparison", "✅", "Research takeaway"),
]

CHATBOT_PAGE = "Research Chatbot"
ALL_PAGES = [item[0] for item in NAV_ITEMS]


def get_theme_mode():
    return "Streamlit"


def inject_app_style(theme_mode="Streamlit"):
    st.markdown(
        """
        <style>
        :root {
            --dap-bg: var(--background-color);
            --dap-card: var(--secondary-background-color);
            --dap-text: var(--text-color);
            --dap-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
            --dap-border: color-mix(in srgb, var(--text-color) 18%, transparent);
        }

        .stApp {
            background: var(--dap-bg) !important;
            color: var(--dap-text) !important;
        }
        [data-testid="stHeader"], header[data-testid="stHeader"] {
            background: var(--dap-bg) !important;
            color: var(--dap-text) !important;
            box-shadow: none !important;
        }
        [data-testid="stToolbar"], [data-testid="stDecoration"] {
            background: transparent !important;
            color: var(--dap-text) !important;
        }
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 4.8rem;
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {
            color: var(--dap-text);
        }

        /* Sidebar Formatting */
        [data-testid="stSidebar"] {
            background: var(--dap-card) !important;
            border-right: 1px solid var(--dap-border);
        }
        [data-testid="stSidebar"] * { color: var(--dap-text) !important; }
        section[data-testid="stSidebar"] hr { margin: 0.55rem 0 !important; }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.34rem !important; }
        .sidebar-compact-title {
            display: flex;
            align-items: center;
            gap: 0.48rem;
            font-size: 1.08rem;
            line-height: 1.15;
            font-weight: 850;
            margin: 0.15rem 0 0.18rem 0;
        }
        .sidebar-compact-subtitle {
            color: var(--dap-muted) !important;
            font-size: 0.74rem;
            line-height: 1.15;
            margin: 0 0 0.55rem 0;
        }

        /* TAB SIDEBAR */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 2.3rem !important;
            border-radius: 12px !important;
            padding: 0.4rem 0.6rem !important;
            margin: 0 !important;
            text-align: left;
            font-size: 0.88rem !important;
            font-weight: 650;
            display: flex;
            align-items: center;
            border: 1px solid transparent !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button:not([disabled]):hover {
            border-color: color-mix(in srgb, var(--primary-color) 62%, transparent) !important;
            background-color: color-mix(in srgb, var(--text-color) 5%, transparent) !important;
        }
        section[data-testid="stSidebar"] .stButton > button[disabled] {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
            color: white !important;
            opacity: 1 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
            cursor: default !important;
        }
        section[data-testid="stSidebar"] .stButton > button[disabled] * {
            color: white !important;
        }

        /* Nút bấm màn hình chính */
        .block-container .stButton > button,
        .block-container button[data-testid="baseButton-secondary"],
        .block-container button[data-testid="baseButton-primary"] {
            border-radius: 14px !important;
            border: 1px solid var(--dap-border) !important;
            background-color: var(--background-color) !important;
            color: var(--dap-text) !important;
            font-weight: 650 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }

        /* Khung chứa Chatbot Bubble Button */
        .st-key-chat_bubble_button {
            position: fixed !important;
            right: 25px !important;
            bottom: 25px !important;
            z-index: 2147483647 !important;
            width: 75px !important;
            height: 75px !important;
        }
        .st-key-chat_bubble_button button {
            width: 100% !important;
            height: 100% !important;
            min-height: 75px !important;
            border-radius: 50% !important;
            padding: 0 !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
            border: none !important;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
            font-weight: 800 !important;
            font-size: 12px !important;
            line-height: 1.2 !important;
        }
        .st-key-chat_bubble_button button:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.6) !important;
        }

        /* CHATBOT TÁI CẤU TRÚC: NỀN ĐẶC */
        div.st-key-floating_chat_panel {
            position: fixed !important;
            bottom: 95px !important;
            right: 20px !important;
            width: 360px !important; 
            height: 75vh !important; 
            background-color: #0E1117 !important; 
            background-image: none !important;
            backdrop-filter: blur(0) !important; 
            border: 1px solid #30363D !important;
            border-radius: 14px !important;
            z-index: 9999999 !important;
            padding: 12px !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.95) !important; 
        }

        div.st-key-floating_chat_panel > div,
        div.st-key-floating_chat_panel [data-testid="stVerticalBlock"],
        div.st-key-floating_chat_panel [data-testid="stVerticalBlockBorderWrapper"],
        div.st-key-floating_chat_panel [data-testid="stForm"] {
            background-color: transparent !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            gap: 0 !important;
            border: none !important;
        }

        .dap-custom-chat-history {
            height: calc(75vh - 105px); 
            overflow-y: auto;
            display: flex;
            flex-direction: column-reverse; 
            gap: 12px;
            padding-right: 6px;
            margin-top: 8px;
            margin-bottom: 8px;
        }
        .dap-custom-chat-history::-webkit-scrollbar { width: 5px; }
        .dap-custom-chat-history::-webkit-scrollbar-thumb { background-color: #4A5568; border-radius: 4px; }

        .dap-msg-row {
            display: flex;
            gap: 8px;
            align-items: flex-end;
            width: 100%;
        }
        .dap-msg-row.user-row {
            justify-content: flex-end;
        }
        .dap-avatar {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            flex-shrink: 0;
            margin-bottom: 2px;
        }
        .dap-avatar.asst { background-color: #f97316; color: white; }
        .dap-avatar.user { background-color: #2563eb; color: white; }

        .dap-msg {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.45;
            color: #F8FAFC;
            word-break: break-word;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        .dap-msg.asst {
            background-color: #1E293B; 
            border-bottom-left-radius: 4px; 
        }
        .dap-msg.user {
            background-color: #2563eb; 
            border-bottom-right-radius: 4px;
        }

        div.st-key-floating_chat_panel [data-testid="column"] {
            padding: 0 4px !important; 
        }
        div.st-key-floating_chat_panel .stButton > button,
        div.st-key-floating_chat_panel [data-testid="stFormSubmitButton"] button {
            min-height: 32px !important;
            height: 32px !important;
            padding: 0 10px !important;
            font-size: 13px !important;
            border-radius: 8px !important;
            border: 1px solid #4A5568 !important;
            background-color: #1E293B !important;
            color: white !important;
            margin: 0 !important;
            font-weight: 600 !important;
        }
        div.st-key-floating_chat_panel input {
            min-height: 38px !important;
            height: 38px !important;
            padding: 0 12px !important;
            font-size: 14px !important;
            border-radius: 8px !important;
            border: 1px solid #4A5568 !important;
            background-color: #1E293B !important;
            color: white !important;
            margin: 0 !important;
        }
        div.st-key-floating_chat_panel [data-testid="stForm"] > div {
            gap: 6px !important; 
        }

        @media (max-width: 700px) {
            .st-key-chat_bubble_button {
                right: 15px !important; bottom: 15px !important;
                width: 60px !important; height: 60px !important;
            }
            .st-key-chat_bubble_button button { min-height: 60px !important; font-size: 0.7rem !important; }
            div.st-key-floating_chat_panel {
                right: 10px !important; bottom: 85px !important;
                width: calc(100vw - 20px) !important;
                height: 70vh !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_page(page_name: str):
    st.query_params["page"] = page_name
    st.rerun()


def get_current_page():
    page = st.query_params.get("page", "Overview")
    if isinstance(page, list):
        page = page[0] if page else "Overview"
    return page if page in ALL_PAGES else "Overview"


def render_sidebar_navigation(current_page: str):
    st.sidebar.markdown(
        '<div class="sidebar-compact-title">📦 Dashboard</div>'
        '<div class="sidebar-compact-subtitle">Forecast-driven inventory replenishment</div>',
        unsafe_allow_html=True,
    )

    compact_labels = {
        "Inventory Simulation Results": "Inventory Simulation",
        "Inventory Time-series Explorer": "Inventory Timeline",
    }

    for page_name, icon, caption in NAV_ITEMS:
        display_name = compact_labels.get(page_name, page_name)
        if current_page == page_name:
            st.sidebar.button(f"{icon}  {display_name}", key=f"nav_{page_name}_active", disabled=True)
        else:
            if st.sidebar.button(f"{icon}  {display_name}", key=f"nav_{page_name}"):
                set_page(page_name)

    st.sidebar.markdown("---")
    st.sidebar.caption("AI2013 / DAP391m Group 3")


def render_chatbot_bubble(current_page="Overview"):
    if st.button("💬\nAsk Research", key="chat_bubble_button"):
        st.session_state.chat_open = True
        st.rerun()

def apply_display_filters(data):
    if "test_metrics" in data:
        data["test_metrics"] = data["test_metrics"][data["test_metrics"]["model"].isin(MAIN_FORECAST_MODELS)].copy()
    if "validation_metrics" in data:
        data["validation_metrics"] = data["validation_metrics"][data["validation_metrics"]["model"].isin(MAIN_FORECAST_MODELS)].copy()
    if "policy" in data:
        data["policy"] = data["policy"][~data["policy"]["forecast_model"].isin(EXCLUDED_SIMULATION_MODELS)].copy()
    if "feature_importance" in data:
        imp = data["feature_importance"].copy()
        imp = imp[imp["model"].isin(MAIN_FORECAST_MODELS)]
        imp = imp[~imp["feature"].astype(str).str.lower().isin(EXCLUDED_FEATURES)]
        data["feature_importance"] = imp.copy()
    return data

@st.cache_data
def load_csv(name: str, parse_dates=None, nrows=None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=parse_dates, nrows=nrows)

def find_data_file(name, folders, aliases=()):
    for folder in folders:
        for file_name in (name, *aliases):
            path = folder / file_name
            if path.exists():
                return path
    checked = [str(folder / file_name) for folder in folders for file_name in (name, *aliases)]
    raise FileNotFoundError(f"Missing {name}. Checked: {checked}")

def load_csv_from(name, folders, aliases=(), parse_dates=None, nrows=None):
    path = find_data_file(name, folders, aliases)
    return pd.read_csv(path, parse_dates=parse_dates, nrows=nrows)

def load_json_from(name, folders, aliases=()):
    path = find_data_file(name, folders, aliases)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

APP_DATA_FOLDERS = [DATA_DIR, PROJECT_DIR / "streamlit_app" / "data"]
FINAL_FOLDERS = [DATA_DIR, PROJECT_DIR / "outputs" / "final_experiment"]
FORECAST_FOLDERS = [DATA_DIR, PROJECT_DIR / "outputs" / "forecasts"]
METRIC_FOLDERS = [DATA_DIR, PROJECT_DIR / "outputs" / "metrics"]

def keep_main_models(df):
    if "model" in df.columns:
        return df[df["model"].isin(MAIN_FORECAST_MODELS)].copy()
    return df

def load_metric_table(name, aliases=()):
    df = load_csv_from(name, METRIC_FOLDERS + FINAL_FOLDERS + APP_DATA_FOLDERS, aliases=aliases)
    df = keep_main_models(df)
    if "weighted_RMSSE_bottom" not in df.columns and "RMSSE" in df.columns:
        df["weighted_RMSSE_bottom"] = df["RMSSE"]
    return df

def load_forecast_timeseries():
    try:
        return load_csv_from("forecast_timeseries.csv", FORECAST_FOLDERS + APP_DATA_FOLDERS, parse_dates=["date"])
    except FileNotFoundError:
        frames = []
        for filename, split in [("validation_forecasts.csv", "Validation"), ("test_forecasts.csv", "Test")]:
            frame = load_csv_from(filename, FORECAST_FOLDERS, parse_dates=["date"])
            if "split" not in frame.columns:
                frame["split"] = split
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

def adapt_policy_table(df):
    out = df.copy()
    if "policy_label" not in out.columns and "forecast_model" in out.columns:
        out["policy_label"] = out["forecast_model"]
    if "forecast_model" not in out.columns:
        out["forecast_model"] = out["policy_label"]
    if "fill_rate" not in out.columns and "service_level_achieved" in out.columns:
        out["fill_rate"] = out["service_level_achieved"]
    if "service_level_achieved" not in out.columns and "fill_rate" in out.columns:
        out["service_level_achieved"] = out["fill_rate"]
    if "relative_cost_savings_pct" not in out.columns and "relative_cost_savings_vs_fixed_pct" in out.columns:
        out["relative_cost_savings_pct"] = out["relative_cost_savings_vs_fixed_pct"]
    if "scenario" not in out.columns:
        out["scenario"] = "base_case"
    return out

def adapt_inventory_table(df):
    out = df.copy()
    if "forecast_model" not in out.columns and "policy_label" in out.columns:
        out["forecast_model"] = out["policy_label"]
    if "scenario" not in out.columns:
        out["scenario"] = "base_case"
    return out

def load_group_counts():
    try:
        return load_csv_from("demand_group_counts.csv", FINAL_FOLDERS + APP_DATA_FOLDERS)
    except FileNotFoundError:
        groups = load_csv_from("training_only_demand_groups.csv", FINAL_FOLDERS)
        return groups.groupby("demand_group", as_index=False)["id"].nunique().rename(columns={"id": "series_count"})

@st.cache_data
def load_all_data():
    policy = adapt_policy_table(load_csv_from("final_policy_comparison.csv", FINAL_FOLDERS + APP_DATA_FOLDERS, aliases=("policy_comparison.csv",)))
    inventory_ts = adapt_inventory_table(
        load_csv_from(
            "inventory_daily_sample.csv",
            FINAL_FOLDERS + APP_DATA_FOLDERS,
            aliases=("inventory_timeseries_sample.csv", "final_inventory_daily_results.csv"),
            parse_dates=["date"],
            nrows=200_000,
        )
    )
    data = {
        "summary": load_json_from("final_experiment_summary.json", FINAL_FOLDERS + APP_DATA_FOLDERS, aliases=("app_data_summary.json",)),
        "test_metrics": load_metric_table("test_metrics_weighted.csv", aliases=("test_metrics.csv", "forecast_metrics_main_models.csv")),
        "validation_metrics": load_metric_table("validation_metrics_weighted.csv", aliases=("validation_metrics.csv",)),
        "policy": policy,
        "feature_importance": load_csv_from("feature_importance.csv", METRIC_FOLDERS + APP_DATA_FOLDERS),
        "hyperparams": load_csv_from("lightgbm_manual_tuning_results.csv", METRIC_FOLDERS + APP_DATA_FOLDERS, aliases=("lightgbm_hyperparameter_comparison.csv",)),
        "forecast_ts": load_forecast_timeseries(),
        "inventory_ts": inventory_ts,
        "segment_params": load_csv_from("selected_segment_policy_parameters.csv", FINAL_FOLDERS + APP_DATA_FOLDERS),
        "group_policy": load_csv_from("final_group_policy_comparison.csv", FINAL_FOLDERS + APP_DATA_FOLDERS),
        "stats": load_csv_from("global_vs_segment_policy_statistics.csv", FINAL_FOLDERS + APP_DATA_FOLDERS),
        "group_counts": load_group_counts(),
        "profiles": load_csv_from("validation_inventory_profiles.csv", FINAL_FOLDERS + APP_DATA_FOLDERS),
    }
    data["forecast_ts"]["date"] = pd.to_datetime(data["forecast_ts"]["date"])
    data["inventory_ts"]["date"] = pd.to_datetime(data["inventory_ts"]["date"])
    data = apply_display_filters(data)
    return data

def format_currency(value):
    return f"{value:,.2f}"

def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def show_plotly(fig):
    st.plotly_chart(style_plotly(fig), width="stretch", theme="streamlit")

def clean_scenario(df):
    out = df.copy()
    out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS).fillna(out["scenario"])
    return out

def show_kpis(policy_df, scenario):
    selected = policy_df[policy_df["scenario"] == scenario]
    if selected.empty:
        st.warning("No policy rows are available for this scenario.")
        return
    best_cost_row = selected.loc[selected["total_cost"].idxmin()]
    best_service_row = selected.loc[selected["service_level_achieved"].idxmax()]
    tuned = selected[selected["forecast_model"].isin(["LightGBM Tuned", "LightGBM Tuned ROP"])]
    fixed = selected[selected["forecast_model"].isin(["Fixed Rule", "Fixed rule"])]

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
        This Streamlit app demonstrates the enhanced research scope: forecast-based, segment-aware replenishment simulation using saved M5 retail sales outputs.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.info("**Dataset**\n\nWalmart M5 sales signals")
    c2.info("**Forecasting models**\n\nSeasonal Naive 28, Moving Average 28, XGBoost, LightGBM, LightGBM Tuned")
    c3.info("**Inventory policy**\n\nControlled reorder-point simulation, not operational stock data")

    st.subheader("Research workflow")
    st.markdown(
        """
        **Sales data** → **Feature engineering** → **Forecasting models** → **Reorder point simulation** → **Cost comparison**
        """
    )

    workflow_path = ASSET_DIR / "fig_00_forecast_to_replenishment_pipeline.png"
    if workflow_path.exists():
        st.image(str(workflow_path), caption="End-to-end research workflow: data preparation, model development, and inventory operations")

    st.subheader("Main dashboard outputs")
    st.write(
        "The app uses processed result files for metrics, sampled forecasts, and inventory simulations. This keeps deployment fast and avoids loading the full raw M5 dataset."
    )

def forecasting_page(data):
    st.title("📈 Forecasting Performance")
    split = st.radio("Select evaluation split", ["Test", "Validation"], horizontal=True)
    metrics = data["test_metrics"] if split == "Test" else data["validation_metrics"]

    st.subheader(f"{split} metrics")
    st.caption("Enhanced scope uses five main forecast models: Seasonal Naive 28, Moving Average 28, XGBoost, LightGBM, and LightGBM Tuned.")
    st.dataframe(metrics, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics, x="model", y="RMSSE", title="RMSSE by model", text_auto=".3f")
        fig.update_layout(xaxis_title="Model", yaxis_title="RMSSE")
        show_plotly(fig)
    with c2:
        fig = px.bar(metrics, x="model", y="weighted_RMSSE_bottom", title="Weighted RMSSE by model", text_auto=".3f")
        fig.update_layout(xaxis_title="Model", yaxis_title="Weighted RMSSE")
        show_plotly(fig)

    best = metrics.sort_values("RMSSE").iloc[0]
    st.success(f"Best RMSSE on the {split.lower()} split: {best['model']} ({best['RMSSE']:.4f}).")

    st.subheader("LightGBM tuning explanation")
    st.dataframe(data["hyperparams"], width="stretch", hide_index=True)

    st.subheader("Feature importance")
    imp = data["feature_importance"].copy()
    model_choices = sorted(imp["model"].unique())
    model = st.selectbox("Select model", model_choices)
    top_n = st.slider("Number of features", min_value=5, max_value=30, value=15)
    imp_plot = imp[imp["model"] == model].sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(imp_plot.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top {top_n} features: {model}")
    show_plotly(fig)

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
    show_plotly(fig)

    st.dataframe(plot_df, width="stretch", hide_index=True)

def inventory_results_page(data):
    st.title("🏬 Inventory Simulation Results")
    st.caption("Inventory values are controlled simulation outputs under saved policy assumptions, not real operating inventory records.")
    policy = clean_scenario(data["policy"])
    scenario_labels = [SCENARIO_LABELS.get(key, key) for key in sorted(policy["scenario"].dropna().unique())]
    label_to_key = {SCENARIO_LABELS.get(key, key): key for key in sorted(policy["scenario"].dropna().unique())}
    default_index = scenario_labels.index("Base case") if "Base case" in scenario_labels else 0
    selected_label = st.selectbox("Scenario", scenario_labels, index=default_index)
    scenario = label_to_key[selected_label]

    show_kpis(policy, scenario)

    filtered = policy[policy["scenario"] == scenario].sort_values("total_cost")
    st.subheader("Policy comparison table")
    st.dataframe(filtered, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(filtered, x="forecast_model", y="total_cost", title="Total cost by model", text_auto=".2s")
        fig.update_layout(xaxis_title="Forecast model", yaxis_title="Total cost")
        show_plotly(fig)
    with c2:
        cost_cols = ["holding_cost", "ordering_cost", "stockout_cost"]
        stacked = filtered[["forecast_model"] + cost_cols].melt("forecast_model", var_name="Cost type", value_name="Cost")
        fig = px.bar(stacked, x="forecast_model", y="Cost", color="Cost type", title="Cost components by model")
        show_plotly(fig)

    st.subheader("Cost across lead-time scenarios")
    fig = px.bar(policy, x="scenario_label", y="total_cost", color="forecast_model", barmode="group", title="Total cost across scenarios")
    fig.update_layout(xaxis_title="Scenario", yaxis_title="Total cost")
    show_plotly(fig)

def segment_analysis_page(data):
    st.title("🧩 Segment Analysis")
    counts = data["group_counts"].copy()
    params = data["segment_params"].copy()
    group_policy = data["group_policy"].copy()
    stats = data["stats"].copy()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Demand group counts")
        st.dataframe(counts, width="stretch", hide_index=True)
        fig = px.pie(counts, names="demand_group", values="series_count", title="Series distribution")
        show_plotly(fig)

    with c2:
        st.subheader("Selected segment-aware parameters")
        st.dataframe(params, width="stretch", hide_index=True)
        fig = px.scatter(
            params,
            x="coverage_days",
            y="safety_multiplier",
            size="validation_total_cost",
            color="demand_group",
            hover_data=["validation_fill_rate", "validation_lost_sales"],
            title="Validation-selected coverage and safety settings",
        )
        show_plotly(fig)

    st.subheader("Global LightGBM vs Segment-Aware LightGBM")
    compare = group_policy[group_policy["policy_label"].isin(["Global LightGBM ROP", "Segment-Aware LightGBM ROP"])].copy()
    st.dataframe(compare, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            compare,
            x="demand_group",
            y="cost_per_demand_unit",
            color="policy_label",
            barmode="group",
            title="Cost per demand unit by segment",
        )
        show_plotly(fig)
    with c2:
        fig = px.line(compare, x="demand_group", y="fill_rate", color="policy_label", markers=True, title="Fill rate by segment")
        fig.add_hline(y=0.95, line_dash="dash", line_color="gray")
        show_plotly(fig)

    if not stats.empty:
        row = stats.iloc[0]
        st.info(
            f"Paired item-store test: mean cost difference = {row['mean_cost_difference']:.4f}, "
            f"median = {row['median_cost_difference']:.4f}, improved series = {row['share_series_improved']:.2%}, "
            f"Wilcoxon p-value = {row['wilcoxon_p_value']:.6e}."
        )

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
    show_plotly(fig)

    c1, c2 = st.columns(2)
    with c1:
        demand_cols = [c for c in ["actual_demand", "selected_forecast"] if c in plot_df.columns]
        dlong = plot_df[["date"] + demand_cols].melt("date", var_name="Series", value_name="Demand")
        fig = px.line(dlong, x="date", y="Demand", color="Series", markers=True, title="Demand and selected forecast")
        show_plotly(fig)
    with c2:
        fig = px.bar(plot_df, x="date", y="order_quantity", title="Order quantity over time")
        show_plotly(fig)

    st.dataframe(plot_df, width="stretch", hide_index=True)

def forecast_window_sum(values, start_idx, days):
    window = np.clip(values[start_idx : start_idx + days], 0, None)
    if len(window) >= days:
        return float(window.sum())
    if len(window) == 0:
        return 0.0
    return float(window.sum() + window.mean() * (days - len(window)))

def target_fill_z(data):
    target = float(data.get("summary", {}).get("base_scenario", {}).get("target_fill_rate", 0.95))
    target = min(max(target, 0.001), 0.999)
    return float(NormalDist().inv_cdf(target))

def policy_recommendation_params(data, demand_group, policy_label):
    base = data.get("summary", {}).get("base_scenario", {})
    coverage_days = int(base.get("coverage_days", 7))
    safety_multiplier = 1.0
    if policy_label == "Segment-Aware LightGBM ROP":
        params = data["segment_params"].set_index("demand_group")
        if demand_group in params.index:
            row = params.loc[demand_group]
            coverage_days = int(row["coverage_days"])
            safety_multiplier = float(row["safety_multiplier"])
    return coverage_days, safety_multiplier

def calculate_order_recommendation(series_df, profile, data, policy_label, model_label, selected_date, lead_time, coverage_days, safety_multiplier, on_hand, incoming):
    ordered = series_df.sort_values("date").reset_index(drop=True)
    start_idx = int(ordered.index[ordered["date"] == selected_date][0])
    z = target_fill_z(data)

    if policy_label == "Fixed rule":
        mean = max(0.0, float(profile.get("validation_mean_demand", 0.0)))
        std = max(0.0, float(profile.get("validation_std_demand", 0.0)))
        lead_forecast = mean * lead_time
        target_forecast = mean * (lead_time + coverage_days)
        safety_stock = z * std * math.sqrt(lead_time) * safety_multiplier
    else:
        pred_col = PREDICTION_COLUMNS[model_label]
        ss_col = SAFETY_COLUMNS[model_label]
        values = ordered[pred_col].astype(float).to_numpy()
        lead_forecast = forecast_window_sum(values, start_idx, lead_time)
        target_forecast = forecast_window_sum(values, start_idx, lead_time + coverage_days)
        safety_stock = max(0.0, float(profile.get(ss_col, 0.0))) * safety_multiplier

    reorder_point = lead_forecast + safety_stock
    target_inventory = target_forecast + safety_stock
    inventory_position = on_hand + incoming
    suggested_order = max(0, math.ceil(target_inventory - inventory_position)) if inventory_position <= reorder_point else 0
    return {
        "lead_forecast": lead_forecast,
        "target_forecast": target_forecast,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_inventory": target_inventory,
        "inventory_position": inventory_position,
        "suggested_order": suggested_order,
        "should_order": inventory_position <= reorder_point,
    }

def what_if_page(data):
    st.title("🧪 Order Recommendation")
    st.markdown(
        "This calculator uses saved forecast CSVs and validation safety-stock profiles. It does not retrain models or require realtime inference."
    )

    forecast = data["forecast_ts"].copy()
    profiles = data["profiles"].copy()
    policy = data["policy"].copy()
    base = data.get("summary", {}).get("base_scenario", {})
    forecast = forecast.drop(columns=["demand_group"], errors="ignore").merge(
        profiles[["id", "demand_group"]].drop_duplicates(),
        on="id",
        how="left",
    )

    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Split", sorted(forecast["split"].dropna().unique()))
    split_df = forecast[forecast["split"] == split]
    group_options = ["All groups"] + sorted(split_df["demand_group"].dropna().unique())
    selected_group = c2.selectbox("Demand group", group_options)
    if selected_group != "All groups":
        split_df = split_df[split_df["demand_group"] == selected_group]
    store = c3.selectbox("Store", sorted(split_df["store_id"].dropna().unique()))
    store_df = split_df[split_df["store_id"] == store]
    labels = (
        store_df[["id", "item_id", "store_id", "demand_group"]]
        .drop_duplicates()
        .assign(label=lambda d: d["item_id"] + " | " + d["store_id"] + " | " + d["demand_group"] + " | " + d["id"])
        .sort_values("label")
    )
    item_label = st.selectbox("Item-store series", labels["label"].tolist())
    item = labels.loc[labels["label"] == item_label, "id"].iloc[0]

    series_df = store_df[store_df["id"] == item].sort_values("date").copy()
    profile_match = profiles[profiles["id"] == item]
    if profile_match.empty:
        st.error("No validation inventory profile is available for this item-store series.")
        return
    profile = profile_match.iloc[0]
    demand_group = str(profile["demand_group"])

    policies = policy["policy_label"].dropna().astype(str).tolist()
    c1, c2, c3, c4 = st.columns(4)
    selected_date = pd.Timestamp(c1.selectbox("Decision date", series_df["date"].dt.date.tolist()))
    policy_label = c2.selectbox("Policy", policies, index=policies.index("Segment-Aware LightGBM ROP") if "Segment-Aware LightGBM ROP" in policies else 0)
    default_model = POLICY_MODEL_MAP.get(policy_label) or "LightGBM"
    model_label = c3.selectbox("Forecast model", MAIN_FORECAST_MODELS, index=MAIN_FORECAST_MODELS.index(default_model))
    lead_time = int(c4.number_input("Lead time", min_value=1, max_value=28, value=int(base.get("lead_time", 7)), step=1))

    coverage_days, safety_multiplier = policy_recommendation_params(data, demand_group, policy_label)
    c1, c2, c3, c4 = st.columns(4)
    on_hand = float(c1.number_input("On-hand inventory", min_value=0.0, value=10.0, step=1.0))
    incoming = float(c2.number_input("Incoming orders", min_value=0.0, value=0.0, step=1.0))
    c3.metric("Demand group", demand_group)
    c4.metric("Policy params", f"C={coverage_days}, k={safety_multiplier:.2f}")

    rec = calculate_order_recommendation(
        series_df,
        profile,
        data,
        policy_label,
        model_label,
        selected_date,
        lead_time,
        coverage_days,
        safety_multiplier,
        on_hand,
        incoming,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory position", f"{rec['inventory_position']:.2f}")
    c2.metric("Reorder point", f"{rec['reorder_point']:.2f}")
    c3.metric("Target inventory", f"{rec['target_inventory']:.2f}")
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
    st.dataframe(detail, width="stretch", hide_index=True)

    pred_col = PREDICTION_COLUMNS[model_label]
    plot_df = series_df[["date", "sales", pred_col]].rename(columns={"sales": "Actual demand", pred_col: f"{model_label} forecast"})
    long = plot_df.melt("date", var_name="Series", value_name="Demand")
    fig = px.line(long, x="date", y="Demand", color="Series", markers=True, title=f"Actual demand and forecast: {item}")
    fig.add_vline(x=selected_date, line_dash="dash", line_color="gray")
    show_plotly(fig)


def summarize_research_context(data):
    metrics = data.get("test_metrics", pd.DataFrame()).copy()
    policy = data.get("policy", pd.DataFrame()).copy()

    context = {
        "forecast_models": sorted(metrics["model"].dropna().unique().tolist()) if "model" in metrics.columns else [],
        "simulation_models": sorted(policy["forecast_model"].dropna().unique().tolist()) if "forecast_model" in policy.columns else [],
        "best_forecast_model": "N/A",
        "best_forecast_rmsse": None,
        "best_cost_model": "N/A",
        "best_cost_value": None,
        "lead_time_scenarios": [],
    }

    if not metrics.empty and "RMSSE" in metrics.columns:
        best = metrics.sort_values("RMSSE").iloc[0]
        context["best_forecast_model"] = best.get("model", "N/A")
        context["best_forecast_rmsse"] = best.get("RMSSE", None)

    if not policy.empty and "total_cost" in policy.columns:
        best_cost = policy.sort_values("total_cost").iloc[0]
        context["best_cost_model"] = best_cost.get("forecast_model", "N/A")
        context["best_cost_value"] = best_cost.get("total_cost", None)
        if "scenario" in policy.columns:
            context["lead_time_scenarios"] = sorted(policy["scenario"].dropna().unique().tolist())

    return context


def is_vietnamese_question(text: str) -> bool:
    q = text.lower()
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    vietnamese_keywords = [
        "dữ liệu", "du lieu", "mô hình", "mo hinh", "dự báo", "du bao", "tồn kho", "ton kho",
        "chi phí", "chi phi", "kết luận", "ket luan", "tốt nhất", "tot nhat", "mục tiêu", "muc tieu",
        "đề tài", "de tai", "là gì", "la gi", "vì sao", "vi sao", "tại sao", "tai sao",
        "thầy", "bài", "nghiên cứu", "nghien cuu", "mô phỏng", "mo phong"
    ]
    return any(ch in q for ch in vietnamese_chars) or any(k in q for k in vietnamese_keywords)


def get_rule_based_response(question, data):
    q = question.lower().strip()
    vi = is_vietnamese_question(question)
    ctx = summarize_research_context(data)

    forecast_models = ", ".join(ctx["forecast_models"]) if ctx["forecast_models"] else "the forecasting models shown in the dashboard"
    simulation_models = ", ".join(ctx["simulation_models"]) if ctx["simulation_models"] else "the replenishment policies shown in the dashboard"
    best_rmsse = ctx["best_forecast_rmsse"]
    best_rmsse_text = f" with RMSSE = {best_rmsse:.4f}" if isinstance(best_rmsse, (int, float, np.floating)) else ""
    best_rmsse_text_vi = f" với RMSSE = {best_rmsse:.4f}" if isinstance(best_rmsse, (int, float, np.floating)) else ""
    best_cost = ctx["best_cost_value"]
    best_cost_text = f" with total cost = {best_cost:,.2f}" if isinstance(best_cost, (int, float, np.floating)) else ""
    best_cost_text_vi = f" với total cost = {best_cost:,.2f}" if isinstance(best_cost, (int, float, np.floating)) else ""

    if any(k in q for k in ["dataset", "data", "m5", "walmart", "dữ liệu", "du lieu"]):
        if vi:
            return (
                "Nghiên cứu này sử dụng bối cảnh dữ liệu bán lẻ Walmart M5. Bộ dữ liệu gốc có doanh số hằng ngày theo sản phẩm, cửa hàng, department và category. "
                "Để deploy nhanh trên Streamlit, app không tải toàn bộ raw data lớn mà dùng các file output đã xử lý như forecasting metrics, actual-vs-forecast sample và inventory simulation results."
            )
        return (
            "This project uses the Walmart M5 retail sales setting. The original M5 data contains daily sales signals across products, stores, departments, and categories. "
            "For deployment, this app does not load the full raw dataset. It uses processed output files, including forecasting metrics, sample actual-vs-forecast series, and inventory simulation results."
        )

    if any(k in q for k in ["model", "forecast", "dự báo", "du bao", "mô hình", "mo hinh", "lightgbm", "xgboost", "naive", "moving average"]):
        if vi:
            return (
                f"Phần forecasting của app so sánh các mô hình: {forecast_models}. "
                f"Dựa trên test RMSSE đang hiển thị, mô hình dự báo tốt nhất là {ctx['best_forecast_model']}{best_rmsse_text_vi}. "
                "Các kết quả này được đọc trực tiếp từ CSV output đã lưu, không phải realtime inference."
            )
        return (
            f"The forecasting comparison in this app includes: {forecast_models}. "
            f"Based on the displayed test RMSSE, the best forecasting model is {ctx['best_forecast_model']}{best_rmsse_text}. "
            "These results are read directly from saved output CSVs, not from realtime inference."
        )

    if any(k in q for k in ["snap"]):
        if vi:
            return (
                "Feature SNAP đã được loại khỏi phần feature importance trong phiên bản app này. "
                "Việc này giúp phần giải thích performance tập trung vào các feature được chọn trong dashboard cuối cùng."
            )
        return (
            "The SNAP feature is excluded from the feature-importance display in this app version. "
            "This keeps the performance interpretation focused on the selected non-SNAP features used in the final dashboard narrative."
        )

    if any(k in q for k in ["simulation", "inventory", "replenishment", "stock", "tồn kho", "ton kho", "mô phỏng", "mo phong"]):
        if vi:
            return (
                f"Phần inventory simulation chuyển forecast thành quyết định bổ sung tồn kho bằng reorder-point logic. Các policy/model trong simulation gồm: {simulation_models}. "
                f"Trong các kết quả đang hiển thị, mức total cost thấp nhất ở một scenario thuộc về {ctx['best_cost_model']}{best_cost_text_vi}."
            )
        return (
            f"The inventory simulation converts forecasts into replenishment decisions using a reorder-point logic. The simulation page focuses on: {simulation_models}. "
            f"Across the displayed simulation results, the lowest single-scenario total cost is achieved by {ctx['best_cost_model']}{best_cost_text}."
        )

    if any(k in q for k in ["segment", "demand group", "group", "nhom", "nhóm"]):
        counts = data.get("group_counts", pd.DataFrame())
        params = data.get("segment_params", pd.DataFrame())
        count_text = ", ".join(f"{row['demand_group']}={int(row['series_count'])}" for _, row in counts.iterrows())
        param_text = "; ".join(
            f"{row['demand_group']}: C={int(row['coverage_days'])}, k={float(row['safety_multiplier']):.2f}"
            for _, row in params.iterrows()
        )
        if vi:
            return f"Demand group counts từ CSV: {count_text}. Tham số segment-aware đã chọn: {param_text}."
        return f"Demand-group counts from CSV: {count_text}. Selected segment-aware parameters: {param_text}."

    if any(k in q for k in ["lead time", "leadtime", "lead-time", "thời gian giao", "thoi gian giao"]):
        if vi:
            return (
                "Lead time là khoảng thời gian từ lúc đặt hàng bổ sung đến lúc hàng về kho. "
                "Trong app, các lead-time scenario giúp kiểm tra xem model dự báo còn hiệu quả không khi việc replenishment chậm hơn hoặc rủi ro hơn."
            )
        return (
            "Lead time is the delay between placing a replenishment order and receiving inventory. "
            "In this research app, lead-time scenarios are used to test whether a forecasting model remains useful when replenishment becomes slower or riskier."
        )

    if any(k in q for k in ["total cost", "cost", "holding", "stockout", "ordering", "chi phí", "chi phi"]):
        if vi:
            return (
                "Total inventory cost gồm các thành phần như holding cost, ordering cost và stockout cost. "
                "Holding cost tăng khi giữ tồn kho quá cao, còn stockout cost tăng khi không đáp ứng đủ nhu cầu. "
                "Vì vậy, model có forecasting metric tốt nhất chưa chắc luôn tạo ra inventory cost thấp nhất."
            )
        return (
            "Total inventory cost combines cost components such as holding cost, ordering cost, and stockout cost. "
            "Holding cost increases when inventory is kept too high, while stockout cost increases when demand cannot be satisfied. "
            "This is why the best forecasting metric does not always produce the lowest inventory cost."
        )

    if any(k in q for k in ["rmsse", "wrmsse", "weighted rmsse", "metric", "metrics", "chỉ số", "chi so"]):
        if vi:
            return (
                "RMSSE và weighted RMSSE là các chỉ số đánh giá forecasting có xét đến scale của từng chuỗi bán hàng. "
                "Giá trị càng thấp thì forecast càng tốt. Dashboard dùng các chỉ số này để so sánh model trước khi đánh giá tác động xuống inventory simulation."
            )
        return (
            "RMSSE and weighted RMSSE are scale-aware forecasting metrics used to compare demand series with different sales volumes. "
            "Lower values indicate better forecasting performance. The dashboard uses these metrics to compare models before evaluating their downstream inventory impact."
        )

    if any(k in q for k in ["best", "result", "conclusion", "takeaway", "trade-off", "tradeoff", "kết luận", "ket luan", "tốt nhất", "tot nhat"]):
        if vi:
            return (
                f"Kết luận chính là có trade-off giữa forecasting accuracy và inventory performance. {ctx['best_forecast_model']} cho kết quả forecasting tốt nhất{best_rmsse_text_vi}, "
                f"trong khi {ctx['best_cost_model']} đạt mức inventory cost thấp nhất ở một scenario đang hiển thị{best_cost_text_vi}. "
                "Vì vậy, bài nghiên cứu nhấn mạnh rằng chọn model không nên chỉ dựa vào forecast metric mà còn cần xét chi phí vận hành tồn kho."
            )
        return (
            f"The main finding is a forecasting-inventory trade-off. {ctx['best_forecast_model']} gives the strongest forecasting result{best_rmsse_text}, "
            f"while {ctx['best_cost_model']} achieves the lowest displayed single-scenario inventory cost{best_cost_text}. "
            "Therefore, the project argues that model selection should consider both forecasting accuracy and operational inventory performance."
        )

    if any(k in q for k in ["purpose", "goal", "objective", "research", "đề tài", "de tai", "mục tiêu", "muc tieu", "nghiên cứu", "nghien cuu"]):
        if vi:
            return (
                "Mục tiêu của app là minh họa forecast-driven inventory replenishment. "
                "App cho thấy forecast từ các model machine learning được chuyển thành quyết định bổ sung tồn kho như thế nào và được so sánh bằng cost-based simulation."
            )
        return (
            "The objective of this research app is to demonstrate forecast-driven inventory replenishment. "
            "It shows how sales forecasts from machine-learning models can be translated into inventory decisions and compared using cost-based simulation."
        )

    if vi:
        return (
            "Mình có thể trả lời các câu hỏi về dataset Walmart M5, mô hình dự báo, RMSSE, feature importance, inventory simulation, lead time, total cost và kết luận nghiên cứu. "
            "Bạn có thể hỏi: 'Model nào tốt nhất?', 'Lead time là gì?', hoặc 'Vì sao forecast tốt hơn nhưng cost chưa chắc thấp hơn?'"
        )
    return (
        "I can answer questions about the Walmart M5 dataset, forecasting models, RMSSE metrics, feature importance, inventory simulation, lead time, total cost, and the main research conclusions. "
        "Try asking: 'Which model has the best RMSSE?', 'What is lead time?', or 'Why can better forecasting still have higher inventory cost?'"
    )

def get_chatbot_response(question, data, current_page):
    return get_rule_based_response(question, data)


def process_chat_query(data, current_page):
    return


def render_floating_chatbot(data, current_page):
    default_greeting = "Hi! Ask me about saved CSV metrics, demand segments, inventory simulation, order recommendation, costs, or conclusions."
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [("assistant", default_greeting)]

    with st.container(key="floating_chat_panel"):
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear", key="clear_chat", width="stretch"):
                st.session_state.chat_history = [("assistant", default_greeting)]
                st.rerun()
        with c2:
            if st.button("❌ Close", key="close_chat", width="stretch"):
                st.session_state.chat_open = False
                st.rerun()

        chat_html = '<div class="dap-custom-chat-history">'
        
        for role, msg in reversed(st.session_state.chat_history[-15:]):
            safe_msg = html.escape(msg).replace("\n", "<br>")
            if role == "assistant":
                chat_html += f'<div class="dap-msg-row"><div class="dap-avatar asst">🤖</div><div class="dap-msg asst">{safe_msg}</div></div>'
            else:
                chat_html += f'<div class="dap-msg-row user-row"><div class="dap-msg user">{safe_msg}</div><div class="dap-avatar user">🙂</div></div>'
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

        with st.form("floating_csv_chat_form", clear_on_submit=True):
            user_q = st.text_input("Ask", placeholder="Gõ câu hỏi của bạn...", label_visibility="collapsed")
            submitted = st.form_submit_button("Send", width="stretch")

        if submitted and user_q.strip():
            st.session_state.chat_history.append(("user", user_q.strip()))
            ans = get_chatbot_response(user_q.strip(), data, current_page)
            st.session_state.chat_history.append(("assistant", ans))
            st.rerun()


def conclusion_page(data):
    st.title("✅ Final Comparison and Research Takeaways")
    metrics = data["test_metrics"].copy()
    policy = clean_scenario(data["policy"])

    st.subheader("Forecasting vs inventory trade-off")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Forecast metrics from saved CSV")
        st.dataframe(metrics.sort_values("RMSSE"), width="stretch", hide_index=True)
    with c2:
        st.caption("Inventory policy comparison from controlled simulation CSV")
        st.dataframe(policy.sort_values("total_cost"), width="stretch", hide_index=True)

    best_forecast = metrics.sort_values("RMSSE").iloc[0]
    best_cost = policy.sort_values("total_cost").iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.success(f"Best forecasting model\n\n**{best_forecast['model']}**")
    c2.success(f"Lowest single-scenario cost\n\n**{best_cost['forecast_model']}**")
    c3.info("Practical interpretation\n\n**Segment-aware replenishment optimizes the decision layer, not just forecast accuracy.**")

    st.markdown(
        """
        **Main takeaway:** forecasting accuracy and inventory cost are related, but they are not identical objectives. The enhanced study evaluates five saved forecast models and then converts forecasts into controlled reorder-point simulation. Segment-aware LightGBM ROP achieves the lowest simulated cost in the loaded final experiment while maintaining the service constraint.
        """
    )

    fig_path = ASSET_DIR / "fig_07_policy_cost_fill_tradeoff.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Project visualization output")


def main():
    theme_mode = get_theme_mode()
    inject_app_style(theme_mode)
    data = load_all_data()
    page = get_current_page()
    process_chat_query(data, page)
    render_sidebar_navigation(page)

    if page == "Overview":
        overview_page(data)
    elif page == "Forecasting Performance":
        forecasting_page(data)
    elif page == "Actual vs Forecast":
        forecast_visual_page(data)
    elif page == "Inventory Simulation Results":
        inventory_results_page(data)
    elif page == "Segment Analysis":
        segment_analysis_page(data)
    elif page == "Inventory Time-series Explorer":
        inventory_timeseries_page(data)
    elif page == "Order Recommendation":
        what_if_page(data)
    elif page == "Final Comparison":
        conclusion_page(data)

    if st.session_state.get("chat_open", False):
        render_floating_chatbot(data, page)

    render_chatbot_bubble(page)

if __name__ == "__main__":
    main()
