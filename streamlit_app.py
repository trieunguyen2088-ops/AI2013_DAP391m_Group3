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

        /* Ẩn các nút kỹ thuật dùng để định tuyến */
        [class^="st-key-hidden_nav_"] { display: none !important; }
        .st-key-chat_bubble_button { display: none !important; }

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
        .nav-list { display: flex; flex-direction: column; gap: 0.18rem; margin: 0.3rem 0; }
        .nav-item {
            display: flex;
            align-items: center;
            border-radius: 10px;
            padding: 0 0.58rem;
            height: 2.1rem;
            min-height: 2.1rem;
            box-sizing: border-box;
            font-size: 0.85rem;
            font-weight: 600;
            width: 100%;
            text-decoration: none !important;
            cursor: pointer;
            transition: background 0.15s;
        }
        .nav-current {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white !important;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22);
        }
        .nav-current * { color: white !important; }
        .nav-link {
            color: var(--dap-text) !important;
            background: transparent;
        }
        .nav-link:hover {
            background: color-mix(in srgb, var(--primary-color) 10%, transparent);
            color: var(--dap-text) !important;
        }

        /* Buttons */
        .stButton > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {
            border-radius: 14px !important;
            border: 1px solid var(--dap-border) !important;
            background: var(--dap-chat-input) !important;
            background-color: var(--dap-chat-input) !important;
            color: var(--dap-text) !important;
            font-weight: 650 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }
        .stButton > button * { color: var(--dap-text) !important; }
        .stButton > button:hover {
            border-color: color-mix(in srgb, var(--primary-color) 62%, transparent) !important;
            box-shadow: 0 8px 18px color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
        }

        /* Cards / workflow */
        .workflow-card {
            border-radius: 22px;
            padding: 1.2rem;
            background: var(--dap-card) !important;
            border: 1px solid var(--dap-border);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
            display: inline-block;
            width: 100%;
        }
        .workflow-card img { display: block; margin: 0 auto; }

        /* Draggable chat bubble */
        #dap-drag-bubble {
            position: fixed;
            right: 26px;
            bottom: 26px;
            width: 72px;
            height: 72px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: #fff;
            font-size: 0.72rem;
            font-weight: 800;
            text-align: center;
            line-height: 1.1;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 2147483647;
            box-shadow: 0 12px 28px rgba(37,99,235,0.38), 0 4px 10px rgba(15,23,42,0.18);
            border: 2px solid rgba(255,255,255,0.25);
            user-select: none;
            transition: box-shadow 0.15s, transform 0.15s;
        }
        #dap-drag-bubble:hover { transform: scale(1.07); }

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
        }
        @media (prefers-color-scheme: dark) {
            .st-key-floating_chat_panel {
                background: #1e293b !important;
                color: #f1f5f9 !important;
                border-color: #334155 !important;
            }
        }
        .st-key-floating_chat_panel > div,
        .st-key-floating_chat_panel [data-testid="stVerticalBlock"],
        .st-key-floating_chat_panel [data-testid="stForm"],
        .st-key-floating_chat_panel [data-testid="stForm"] > div {
            background: transparent !important;
        }
        .st-key-floating_chat_panel input,
        .st-key-floating_chat_panel textarea {
            background: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }
        @media (prefers-color-scheme: dark) {
            .st-key-floating_chat_panel input,
            .st-key-floating_chat_panel textarea {
                background: #0f172a !important;
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
            flex: 0 0 22px; width: 22px; height: 22px; border-radius: 6px;
            display: inline-flex; align-items: center; justify-content: center;
            color: white !important; font-size: 0.75rem;
        }
        .dap-chat-avatar.assistant { background: #f97316 !important; }
        .dap-chat-avatar.user { background: #2563eb !important; }
        .dap-chat-message-text {
            font-size: 0.78rem; line-height: 1.4; color: #0f172a !important; word-break: break-word;
        }
        @media (prefers-color-scheme: dark) { .dap-chat-message-text { color: #f1f5f9 !important; } }
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
    compact_labels = {
        "Inventory Simulation Results": "Inventory Simulation",
        "Inventory Time-series Explorer": "Inventory Timeline",
    }

    # Render giao diện Custom HTML của người dùng
    nav_html = '<div class="sidebar-compact-title">📦 Dashboard</div>'
    nav_html += '<div class="sidebar-compact-subtitle">Forecast-driven inventory replenishment</div>'
    nav_html += '<div class="nav-list">'

    for page_name, icon, caption in NAV_ITEMS:
        display_name = compact_labels.get(page_name, page_name)
        safe_id = page_name.replace(" ", "_").replace("-", "_")
        
        if current_page == page_name:
            nav_html += f'<div class="nav-item nav-current" data-page="{safe_id}">{icon} {display_name}</div>'
        else:
            nav_html += f'<div class="nav-item nav-link custom-dap-link" data-page="{safe_id}">{icon} {display_name}</div>'

    nav_html += '</div>'
    nav_html += '<hr style="margin:0.55rem 0; border-color: var(--dap-border);">'
    nav_html += '<div style="font-size:0.74rem; color: var(--dap-muted);">AI2013 / DAP391m Group 3</div>'

    st.sidebar.markdown(nav_html, unsafe_allow_html=True)

    # Khởi tạo các nút ẩn của Streamlit để định tuyến (Routing)
    for page_name, _, _ in NAV_ITEMS:
        safe_id = page_name.replace(" ", "_").replace("-", "_")
        if st.sidebar.button(page_name, key=f"hidden_nav_{safe_id}"):
            set_page(page_name)

    # Chèn JS để map HTML tuỳ chỉnh sang các nút ẩn (Vượt rào chặn onclick của Streamlit)
    js = """
    <script>
    (function bindNav() {
        var doc = window.parent.document || document;
        var links = doc.querySelectorAll('.custom-dap-link');
        if (links.length === 0) { setTimeout(bindNav, 200); return; }
        
        links.forEach(function(link) {
            if (link.dataset.bound) return;
            link.dataset.bound = "true";
            link.addEventListener('click', function() {
                var pageId = this.getAttribute('data-page');
                var hiddenBtn = doc.querySelector('.st-key-hidden_nav_' + pageId + ' button');
                if (hiddenBtn) hiddenBtn.click();
            });
        });
    })();
    </script>
    """
    st.sidebar.markdown(js, unsafe_allow_html=True)


def render_chatbot_bubble(current_page="Overview"):
    """Render a draggable floating chat bubble via JS. Click opens chat, drag moves it."""
    
    # Render giao diện HTML của bong bóng chat
    st.markdown('<div id="dap-drag-bubble">💬<br><span style="font-size:0.62rem">Ask AI</span></div>', unsafe_allow_html=True)
    
    # Nút ẩn thực hiện việc mở chat
    if st.button("💬 Ask Research", key="chat_bubble_button"):
        st.session_state.chat_open = True
        st.rerun()

    # JS xử lý kéo thả và click mở chat
    drag_js = """
    <script>
    (function initChatBubble() {
        var doc = window.parent.document || document;
        var el = doc.getElementById('dap-drag-bubble');
        if (!el) { setTimeout(initChatBubble, 200); return; }
        if (el.dataset.bound) return;
        el.dataset.bound = "true";

        var dragging = false, moved = false;
        var startX, startY, origRight, origBottom;

        function getRight() { return window.innerWidth - el.getBoundingClientRect().right; }
        function getBottom() { return window.innerHeight - el.getBoundingClientRect().bottom; }

        el.addEventListener('mousedown', function(e) {
            dragging = true; moved = false;
            startX = e.clientX; startY = e.clientY;
            origRight = getRight(); origBottom = getBottom();
            e.preventDefault();
        });

        doc.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var dx = e.clientX - startX, dy = e.clientY - startY;
            if (Math.abs(dx) > 4 || Math.abs(dy) > 4) moved = true;
            el.style.right = (origRight - dx) + 'px';
            el.style.bottom = (origBottom - dy) + 'px';
            el.style.left = 'auto'; el.style.top = 'auto';
            syncPanel();
        });

        doc.addEventListener('mouseup', function(e) {
            if (!dragging) return;
            dragging = false;
            if (!moved) {
                var btn = doc.querySelector('.st-key-chat_bubble_button button');
                if (btn) btn.click();
            }
        });

        function syncPanel() {
            var panel = doc.querySelector('.st-key-floating_chat_panel');
            if (!panel) return;
            var r = parseFloat(el.style.right) || 26;
            var b = parseFloat(el.style.bottom) || 26;
            var bh = el.offsetHeight || 72;
            panel.style.right = r + 'px';
            panel.style.bottom = (b + bh + 12) + 'px';
        }
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
        return pd.DataFrame()

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
    """Keep Plotly charts compatible with Streamlit's native theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def show_plotly(fig):
    # theme="streamlit" lets charts follow Streamlit's System / Light / Dark menu.
    st.plotly_chart(style_plotly(fig), use_container_width=True, theme="streamlit")

def clean_scenario(df):
    out = df.copy()
    if not out.empty:
        out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS).fillna(out["scenario"])
    return out

def show_kpis(policy_df, scenario):
    if policy_df.empty: return
    selected = policy_df[policy_df["scenario"] == scenario]
    if selected.empty: return
    
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
    c2.info("**Forecasting models**\n\nNaive, Random Forest, XGBoost, LightGBM, Tuned LightGBM")
    c3.info("**Inventory policy**\n\nForecast-driven reorder point simulation")

    st.subheader("Research workflow")
    st.markdown(
        """
        **Sales data** → **Feature engineering** → **Forecasting models** → **Reorder point simulation** → **Cost comparison**
        """
    )

    workflow_path = ASSET_DIR / "fig2.png"
    if workflow_path.exists():
        st.image(str(workflow_path), caption="End-to-end research workflow: data preparation, model development, and inventory operations")

    st.subheader("Main dashboard outputs")
    st.write(
        "The app uses processed result files for metrics, sampled forecasts, and inventory simulations. This keeps deployment fast and avoids loading the full raw M5 dataset."
    )

def forecasting_page(data):
    st.title("📈 Forecasting Performance")
    if data["test_metrics"].empty: return
    split = st.radio("Select evaluation split", ["Test", "Validation"], horizontal=True)
    metrics = data["test_metrics"] if split == "Test" else data["validation_metrics"]

    st.subheader(f"{split} metrics")
    st.caption("CatBoost and Moving Average are excluded from the app display. The SNAP feature is also excluded from the feature-importance view.")
    st.dataframe(metrics, use_container_width=True)

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

    if not data["hyperparams"].empty:
        st.subheader("LightGBM tuning explanation")
        st.dataframe(data["hyperparams"], use_container_width=True, hide_index=True)

    if not data["feature_importance"].empty:
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
    if df.empty: return

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

    st.dataframe(plot_df, use_container_width=True, hide_index=True)

def inventory_results_page(data):
    st.title("🏬 Inventory Simulation Results")
    st.caption("Simulation focuses on Naive, Fixed Rule, LightGBM, and Tuned LightGBM. XGBoost and Random Forest are excluded from simulation because they are used only for forecasting comparison in this app.")
    policy = clean_scenario(data["policy"])
    if policy.empty: return
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

def inventory_timeseries_page(data):
    st.title("📉 Inventory Time-series Explorer")
    df = data["inventory_ts"].copy()
    if df.empty: return
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


def summarize_research_context(data):
    """Build a compact, data-aware context for the chatbot."""
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


def get_gemini_api_key():
    """Read Gemini API key from Streamlit secrets or environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            return st.secrets["gemini"]["api_key"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def build_gemini_prompt(question, data):
    ctx = summarize_research_context(data)
    best_rmsse = ctx.get("best_forecast_rmsse")
    best_cost = ctx.get("best_cost_value")
    best_rmsse_text = f"{best_rmsse:.4f}" if isinstance(best_rmsse, (int, float, np.floating)) else "N/A"
    best_cost_text = f"{best_cost:,.2f}" if isinstance(best_cost, (int, float, np.floating)) else "N/A"

    return f"""
You are a concise research assistant embedded in a Streamlit dashboard for a student research project.
Answer only about this project: Walmart M5 sales forecasting and forecast-driven inventory replenishment simulation.
If the user asks in Vietnamese, answer in Vietnamese. If the user asks in English, answer in English.
Do not invent exact numbers that are not provided in the context. Be clear when something is not available in the dashboard.
Keep answers short, presentation-friendly, and focused on the research.

Project context:
- Topic: Inventory Replenishment Optimization Using Sales Signals.
- Dataset setting: Walmart M5 retail sales data.
- App uses processed outputs only, not the full raw M5 dataset.
- Forecasting models shown: {', '.join(ctx.get('forecast_models', []))}.
- Simulation models/policies shown: {', '.join(ctx.get('simulation_models', []))}.
- Best displayed forecasting model by test RMSSE: {ctx.get('best_forecast_model')} with RMSSE {best_rmsse_text}.
- Lowest displayed single-scenario inventory cost model/policy: {ctx.get('best_cost_model')} with total cost {best_cost_text}.
- CatBoost and Moving Average are excluded from this app.
- XGBoost and Random Forest are used only in forecasting comparison, not in inventory simulation.
- SNAP is excluded from the feature-importance display.
- Main research takeaway: forecasting accuracy and inventory cost are related but not identical objectives; model selection should consider both prediction metrics and downstream inventory cost.

User question: {question}
"""


def get_gemini_response(question, data):
    """Call Gemini API when a key is configured. Return None if unavailable or failed."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(build_gemini_prompt(question, data))
        answer = getattr(response, "text", "") or ""
        return answer.strip() or None
    except Exception as exc:
        st.session_state["gemini_last_error"] = str(exc)
        return None


def get_rule_based_response(question, data):
    """Fallback rule-based bilingual research assistant for the dashboard."""
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

    if any(k in q for k in ["model", "forecast", "dự báo", "du bao", "mô hình", "mo hinh", "lightgbm", "xgboost", "random forest", "naive"]):
        if vi:
            return (
                f"Phần forecasting của app so sánh các mô hình: {forecast_models}. "
                f"Dựa trên test RMSSE đang hiển thị, mô hình dự báo tốt nhất là {ctx['best_forecast_model']}{best_rmsse_text_vi}. "
                "XGBoost và Random Forest chỉ được giữ để so sánh forecasting, không dùng trong phần inventory simulation."
            )
        return (
            f"The forecasting comparison in this app includes: {forecast_models}. "
            f"Based on the displayed test RMSSE, the best forecasting model is {ctx['best_forecast_model']}{best_rmsse_text}. "
            "XGBoost and Random Forest are kept for forecasting comparison, but they are not used in the inventory simulation page."
        )

    if any(k in q for k in ["catboost", "moving average", "moving", "catboot"]):
        if vi:
            return (
                "CatBoost và Moving Average đã được bỏ khỏi app để dashboard khớp với phạm vi nghiên cứu cuối cùng. "
                "App hiện tập trung vào Seasonal Naive 28, Random Forest, XGBoost, LightGBM và Tuned LightGBM trong phần forecasting comparison."
            )
        return (
            "CatBoost and Moving Average were removed from the app display to keep the final dashboard aligned with the revised research scope. "
            "The app focuses on Seasonal Naive 28, Random Forest, XGBoost, LightGBM, and Tuned LightGBM for forecasting comparison."
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
                "App cho thấy forecast từ các model machine learning được chuyển thành quyết định bổ báo tồn kho như thế nào và được so sánh bằng cost-based simulation."
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


def get_chatbot_response(question, data):
    """Use Gemini when available; fall back to local rule-based responses."""
    gemini_answer = get_gemini_response(question, data)
    if gemini_answer:
        return gemini_answer
    return get_rule_based_response(question, data)


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
            safe_message = html.escape(str(message)).replace("\n", "<br>")
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
    if data["test_metrics"].empty or data["policy"].empty: return
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
    theme_mode = get_theme_mode()
    inject_app_style(theme_mode)
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
