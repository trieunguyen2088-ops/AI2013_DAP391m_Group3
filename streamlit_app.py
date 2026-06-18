from pathlib import Path
import json
import os
from urllib.parse import quote
import html
import uuid

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
    "Random Forest": "random_forest_pred",
    "XGBoost": "xgboost_pred",
    "LightGBM": "lightgbm_pred",
    "LightGBM Tuned": "lightgbm_tuned_pred",
}

EXCLUDED_ALL_MODELS = {"CatBoost", "Moving Average 28"}
EXCLUDED_SIMULATION_MODELS = {"XGBoost", "Random Forest", "CatBoost", "Moving Average 28"}
EXCLUDED_FEATURES = {"snap"}

NAV_ITEMS = [
    ("Overview", "🏠", "Project overview"),
    ("Forecasting Performance", "📈", "Forecast metrics"),
    ("Actual vs Forecast", "🔍", "Demand visualization"),
    ("Inventory Simulation Results", "🏬", "Cost simulation"),
    ("Inventory Time-series Explorer", "📉", "Inventory timeline"),
    ("What-if Simulator", "🧪", "Scenario calculator"),
    ("Final Comparison", "✅", "Research takeaway"),
]

CHATBOT_PAGE = "Research Chatbot"
ALL_PAGES = [item[0] for item in NAV_ITEMS]


def get_theme_mode():
    """Follow Streamlit's built-in System / Light / Dark theme."""
    return "Streamlit"


def inject_app_style(theme_mode="Streamlit"):
    """Clean CSS that follows Streamlit native Light/Dark theme."""
    st.markdown(
        """
        <style>
        :root {
            --dap-bg: var(--background-color);
            --dap-card: var(--secondary-background-color);
            --dap-text: var(--text-color);
            --dap-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
            --dap-border: color-mix(in srgb, var(--text-color) 18%, transparent);
            --dap-chat-bg: var(--secondary-background-color);
            --dap-chat-input: var(--background-color);
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

        /* Sidebar */
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
        
        /* Cải tiến hiển thị Streamlit Sidebar Buttons giống hệt Menu tùy chỉnh */
        [data-testid="stSidebar"] .stButton > button {
            border: none !important;
            border-radius: 10px !important;
            padding: 0.2rem 0.58rem !important;
            height: 2.3rem !important;
            min-height: 2.3rem !important;
            box-shadow: none !important;
            background: transparent !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
            color: var(--dap-text) !important;
        }
        
        /* Highlight trang hiện tại (Dùng class này bằng st.markdown trick nếu cần) */
        .active-nav-btn {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
            color: white !important;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22) !important;
        }

        /* Buttons chung */
        .stButton > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {
            border-radius: 14px !important;
            border: 1px solid var(--dap-border) !important;
            background: var(--dap-chat-input) !important;
            color: var(--dap-text) !important;
            font-weight: 650 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }
        .stButton > button:hover {
            border-color: color-mix(in srgb, var(--primary-color) 62%, transparent) !important;
            box-shadow: 0 8px 18px color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
        }

        /* Fixed Streamlit chatbot panel */
        .st-key-floating_chat_panel {
            position: fixed !important;
            right: 24px !important;
            bottom: 108px !important;
            width: min(320px, calc(100vw - 40px)) !important;
            max-height: calc(100vh - 140px) !important;
            overflow-y: auto !important;
            z-index: 2147483600 !important;
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 18px !important;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.22) !important;
            padding: 0.75rem !important;
            font-size: 0.82rem !important;
            opacity: 1 !important;
            backdrop-filter: none !important;
            isolation: isolate !important;
        }
        @media (prefers-color-scheme: dark) {
            .st-key-floating_chat_panel {
                background: #1e293b !important;
                color: #f1f5f9 !important;
                border-color: #334155 !important;
            }
        }
        .dap-fixed-chat-title {
            font-size: 1rem;
            font-weight: 750;
            margin: 0 0 0.45rem 0;
            color: #0f172a !important;
        }
        @media (prefers-color-scheme: dark) {
            .dap-fixed-chat-title { color: #f1f5f9 !important; }
        }
        .dap-chat-message-row {
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            margin: 0.4rem 0;
            padding: 0.45rem 0.55rem;
            border-radius: 10px;
            background: #f1f5f9 !important;
            border: 1px solid #e2e8f0;
        }
        .dap-chat-message-row.user {
            background: #eff6ff !important;
            border-color: #bfdbfe;
        }
        @media (prefers-color-scheme: dark) {
            .dap-chat-message-row { background: #0f172a !important; border-color: #334155; }
            .dap-chat-message-row.user { background: #1e3a5f !important; border-color: #2563eb; }
        }
        .dap-chat-avatar {
            flex: 0 0 22px;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white !important;
            font-size: 0.75rem;
        }
        .dap-chat-avatar.assistant { background: #f97316 !important; }
        .dap-chat-avatar.user { background: #2563eb !important; }
        .dap-chat-message-text {
            font-size: 0.78rem;
            line-height: 1.4;
            color: #0f172a !important;
            word-break: break-word;
        }
        @media (prefers-color-scheme: dark) {
            .dap-chat-message-text { color: #f1f5f9 !important; }
        }

        /* Ẩn nút gốc của chat_bubble_button */
        .st-key-chat_bubble_button {
            position: fixed !important;
            right: -9999px !important;
            bottom: -9999px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            z-index: -1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_page(page_name: str):
    st.session_state.current_page = page_name
    st.query_params["page"] = page_name
    st.rerun()

def get_current_page():
    if "current_page" not in st.session_state:
        page = st.query_params.get("page", "Overview")
        if isinstance(page, list):
            page = page[0] if page else "Overview"
        st.session_state.current_page = page if page in ALL_PAGES else "Overview"
    return st.session_state.current_page

def render_sidebar_navigation(current_page: str):
    compact_labels = {
        "Inventory Simulation Results": "Inventory Simulation",
        "Inventory Time-series Explorer": "Inventory Timeline",
    }

    st.sidebar.markdown('<div class="sidebar-compact-title">📦 Dashboard</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-compact-subtitle">Forecast-driven inventory replenishment</div>', unsafe_allow_html=True)

    # Sử dụng Native Streamlit Button để không bị lỗi reload trang
    for page_name, icon, caption in NAV_ITEMS:
        display_name = compact_labels.get(page_name, page_name)
        
        # Thêm ký hiệu đặc biệt nếu là trang đang mở
        prefix = "👉 " if current_page == page_name else f"{icon} "
        
        if st.sidebar.button(f"{prefix}{display_name}", key=f"nav_{page_name}", use_container_width=True):
            set_page(page_name)

    st.sidebar.markdown('<hr style="margin:0.55rem 0; border-color: var(--dap-border);">', unsafe_allow_html=True)
    
    # Nút mở Chatbot dự phòng trên Sidebar
    if st.sidebar.button("🤖 Open AI Chatbot", use_container_width=True, type="primary"):
        st.session_state.chat_open = True
        st.rerun()
        
    st.sidebar.markdown('<div style="font-size:0.74rem; color: var(--dap-muted); text-align:center; margin-top:10px;">AI2013 / DAP391m Group 3</div>', unsafe_allow_html=True)


def render_chatbot_bubble(current_page="Overview"):
    """Render a draggable floating chat bubble via JS. Click opens chat, drag moves it."""
    # Nút thật của Streamlit bị ẩn
    if st.button("💬 Ask Research", key="chat_bubble_button"):
        st.session_state.chat_open = True
        st.rerun()

    # Dùng JS để kích hoạt click vào nút ẩn
    drag_js = """
<div id="dap-drag-bubble" style="
    position: fixed; right: 26px; bottom: 26px; width: 72px; height: 72px;
    border-radius: 50%; background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: #fff; font-size: 0.72rem; font-weight: 800; text-align: center; line-height: 1.1;
    display: flex; align-items: center; justify-content: center; cursor: pointer;
    z-index: 2147483647; box-shadow: 0 12px 28px rgba(37,99,235,0.38); border: 2px solid rgba(255,255,255,0.25);
    user-select: none;">
    💬<br><span style="font-size:0.62rem">Ask AI</span>
</div>
<script>
(function() {
    var el = document.getElementById('dap-drag-bubble');
    if (!el) return;
    
    el.addEventListener('click', function(e) {
        var btn = window.parent.document.querySelector('.st-key-chat_bubble_button button') || 
                  document.querySelector('.st-key-chat_bubble_button button');
        if (btn) {
            btn.click();
        }
    });
})();
</script>
"""
    st.markdown(drag_js, unsafe_allow_html=True)

def apply_display_filters(data):
    if "test_metrics" in data:
        data["test_metrics"] = data["test_metrics"][~data["test_metrics"]["model"].isin(EXCLUDED_ALL_MODELS)].copy()
    if "validation_metrics" in data:
        data["validation_metrics"] = data["validation_metrics"][~data["validation_metrics"]["model"].isin(EXCLUDED_ALL_MODELS)].copy()
    if "policy" in data:
        data["policy"] = data["policy"][~data["policy"]["forecast_model"].isin(EXCLUDED_SIMULATION_MODELS)].copy()
    if "feature_importance" in data:
        imp = data["feature_importance"].copy()
        imp = imp[~imp["model"].isin(EXCLUDED_ALL_MODELS)]
        imp = imp[~imp["feature"].astype(str).str.lower().isin(EXCLUDED_FEATURES)]
        data["feature_importance"] = imp.copy()
    return data

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(DATA_DIR / name)
    except FileNotFoundError:
        return pd.DataFrame() # Tránh lỗi sập web nếu thiếu file dữ liệu

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
    if not data["forecast_ts"].empty:
        data["forecast_ts"]["date"] = pd.to_datetime(data["forecast_ts"]["date"])
    if not data["inventory_ts"].empty:
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
    st.plotly_chart(style_plotly(fig), use_container_width=True, theme="streamlit")

def clean_scenario(df):
    if df.empty: return df
    out = df.copy()
    out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS).fillna(out["scenario"])
    return out

def show_kpis(policy_df, scenario):
    if policy_df.empty: return
    selected = policy_df[policy_df["scenario"] == scenario]
    if selected.empty: return
    
    best_cost_row = selected.loc[selected["total_cost"].idxmin()]
    best_service_row = selected.loc[selected["service_level_achieved"].idxmax()]
    tuned = selected[selected["forecast_model"] == "LightGBM Tuned"]
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
    st.markdown("This Streamlit app demonstrates how forecast outputs can be converted into inventory replenishment decisions. It uses the Walmart M5 retail sales setting and compares forecasting performance with downstream inventory simulation cost.")

    c1, c2, c3 = st.columns(3)
    c1.info("**Dataset**\n\nWalmart M5 sales signals")
    c2.info("**Forecasting models**\n\nNaive, Random Forest, XGBoost, LightGBM, Tuned LightGBM")
    c3.info("**Inventory policy**\n\nForecast-driven reorder point simulation")

    st.subheader("Research workflow")
    st.markdown("**Sales data** → **Feature engineering** → **Forecasting models** → **Reorder point simulation** → **Cost comparison**")

    workflow_path = ASSET_DIR / "fig2.png"
    if workflow_path.exists():
        st.image(str(workflow_path), caption="End-to-end research workflow: data preparation, model development, and inventory operations")

def forecasting_page(data):
    st.title("📈 Forecasting Performance")
    if data["test_metrics"].empty:
        st.warning("No data found for metrics.")
        return
        
    split = st.radio("Select evaluation split", ["Test", "Validation"], horizontal=True)
    metrics = data["test_metrics"] if split == "Test" else data["validation_metrics"]

    st.dataframe(metrics, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics, x="model", y="RMSSE", title="RMSSE by model", text_auto=".3f")
        show_plotly(fig)
    with c2:
        fig = px.bar(metrics, x="model", y="weighted_RMSSE_bottom", title="Weighted RMSSE by model", text_auto=".3f")
        show_plotly(fig)

    if "feature_importance" in data and not data["feature_importance"].empty:
        st.subheader("Feature importance")
        imp = data["feature_importance"].copy()
        model = st.selectbox("Select model", sorted(imp["model"].unique()))
        top_n = st.slider("Number of features", 5, 30, 15)
        imp_plot = imp[imp["model"] == model].sort_values("importance", ascending=False).head(top_n)
        fig = px.bar(imp_plot.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top {top_n} features: {model}")
        show_plotly(fig)

def forecast_visual_page(data):
    st.title("🔍 Actual vs Forecast Visualization")
    df = data["forecast_ts"].copy()
    if df.empty: return st.warning("No forecast visualization data available.")

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

def inventory_results_page(data):
    st.title("🏬 Inventory Simulation Results")
    policy = clean_scenario(data["policy"])
    if policy.empty: return st.warning("No inventory policy data available.")
    
    scenario_labels = list(SCENARIO_LABELS.values())
    label_to_key = {v: k for k, v in SCENARIO_LABELS.items()}
    selected_label = st.selectbox("Scenario", scenario_labels, index=scenario_labels.index("Base case"))
    scenario = label_to_key[selected_label]

    show_kpis(policy, scenario)

    filtered = policy[policy["scenario"] == scenario].sort_values("total_cost")
    st.subheader("Policy comparison table")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

def inventory_timeseries_page(data):
    st.title("📉 Inventory Time-series Explorer")
    df = data["inventory_ts"].copy()
    if df.empty: return st.warning("No inventory timeseries data available.")
    
    scenario_labels = {k: SCENARIO_LABELS.get(k, k) for k in df["scenario"].unique()}
    label_to_key = {v: k for k, v in scenario_labels.items()}

    c1, c2, c3 = st.columns(3)
    scenario_label = c1.selectbox("Scenario", sorted(label_to_key.keys()))
    scenario = label_to_key[scenario_label]
    model = c2.selectbox("Forecast model / policy", sorted(df[df["scenario"] == scenario]["forecast_model"].dropna().unique()))
    item = c3.selectbox("Item/store series", sorted(df[(df["scenario"] == scenario) & (df["forecast_model"] == model)]["id"].unique()))

    plot_df = df[(df["scenario"] == scenario) & (df["forecast_model"] == model) & (df["id"] == item)].sort_values("date")

    line_cols = [c for c in ["on_hand_inventory", "reorder_point", "inventory_position"] if c in plot_df.columns]
    long = plot_df[["date"] + line_cols].melt("date", var_name="Series", value_name="Value")
    fig = px.line(long, x="date", y="Value", color="Series", markers=True, title=f"Inventory simulation: {item}")
    show_plotly(fig)

def what_if_page(data):
    st.title("🧪 What-if Replenishment Calculator")
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

    st.write(f"**Order Quantity Suggestion: {order_quantity:.2f}**")

def summarize_research_context(data):
    metrics = data.get("test_metrics", pd.DataFrame()).copy()
    policy = data.get("policy", pd.DataFrame()).copy()

    context = {
        "forecast_models": sorted(metrics["model"].dropna().unique().tolist()) if "model" in metrics.columns else [],
        "simulation_models": sorted(policy["forecast_model"].dropna().unique().tolist()) if "forecast_model" in policy.columns else [],
        "best_forecast_model": "N/A", "best_forecast_rmsse": None,
        "best_cost_model": "N/A", "best_cost_value": None,
    }
    if not metrics.empty and "RMSSE" in metrics.columns:
        best = metrics.sort_values("RMSSE").iloc[0]
        context["best_forecast_model"] = best.get("model", "N/A")
        context["best_forecast_rmsse"] = best.get("RMSSE", None)
    if not policy.empty and "total_cost" in policy.columns:
        best_cost = policy.sort_values("total_cost").iloc[0]
        context["best_cost_model"] = best_cost.get("forecast_model", "N/A")
        context["best_cost_value"] = best_cost.get("total_cost", None)
    return context

def build_gemini_prompt(question, data):
    ctx = summarize_research_context(data)
    best_rmsse_text = f"{ctx.get('best_forecast_rmsse', 0):.4f}"
    return f"""
Project context: Walmart M5 sales forecasting and inventory replenishment simulation.
Forecasting models: {', '.join(ctx.get('forecast_models', []))}.
Simulation models: {', '.join(ctx.get('simulation_models', []))}.
Best test RMSSE: {ctx.get('best_forecast_model')} ({best_rmsse_text}).
User question: {question}
"""

def get_rule_based_response(question, data):
    return "Hi, I'm the fallback rule-based assistant. Please ask about models, RMSSE, or simulation costs!"

def get_chatbot_response(question, data):
    # Bạn có thể giữ code kết nối Gemini ở đây
    return get_rule_based_response(question, data)

def _escape_text(value):
    return html.escape(str(value)).replace("\n", "<br>")

def render_floating_chatbot(data, current_page):
    """Render a solid non-modal Streamlit chat panel fixed in the bottom-right corner."""
    default_greeting = "Hi! Ask me about M5 data, forecasting models, inventory simulation, costs, or conclusions."
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [("assistant", default_greeting)]

    with st.container(key="floating_chat_panel"):
        st.markdown('<div class="dap-fixed-chat-title">Research Assistant Chatbot</div>', unsafe_allow_html=True)
        top_cols = st.columns([1, 1])
        with top_cols[0]:
            if st.button("Clear chat history", key="clear_floating_chat", use_container_width=True):
                st.session_state.chat_history = [("assistant", default_greeting)]
                st.rerun()
        with top_cols[1]:
            if st.button("Close", key="close_floating_chat", use_container_width=True):
                st.session_state.chat_open = False
                st.rerun()

        for role, message in st.session_state.chat_history[-8:]:
            avatar = "🤖" if role == "assistant" else "🙂"
            avatar_class = "assistant" if role == "assistant" else "user"
            safe_message = _escape_text(message)
            st.markdown(
                f'''
                <div class="dap-chat-message-row {role}">
                    <div class="dap-chat-avatar {avatar_class}">{avatar}</div>
                    <div class="dap-chat-message-text">{safe_message}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

        with st.form("floating_gemini_chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question",
                placeholder="Ask about data, models, RMSSE, simulation, or conclusions...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_question.strip():
            st.session_state.chat_history.append(("user", user_question.strip()))
            answer = get_chatbot_response(user_question.strip(), data)
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()

def conclusion_page(data):
    st.title("✅ Final Comparison and Research Takeaways")
    st.write("Conclusion goes here.")

def main():
    inject_app_style()
    data = load_all_data()
    page = get_current_page()
    render_sidebar_navigation(page)

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

    if st.session_state.get("chat_open", False):
        render_floating_chatbot(data, page)

    render_chatbot_bubble(page)

if __name__ == "__main__":
    main()
