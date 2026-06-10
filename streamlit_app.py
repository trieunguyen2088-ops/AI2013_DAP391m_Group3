from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

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
ALL_PAGES = [item[0] for item in NAV_ITEMS] + [CHATBOT_PAGE]


def get_theme_mode():
    """Return the selected visual theme for the Streamlit app."""
    return "Dark" if st.session_state.get("dark_mode", False) else "Light"


def inject_app_style(theme_mode="Light"):
    is_dark = theme_mode == "Dark"
    bg = "#0b1220" if is_dark else "#f8fafc"
    card = "#111827" if is_dark else "#ffffff"
    soft_card = "#1f2937" if is_dark else "#f8fafc"
    text = "#e5e7eb" if is_dark else "#0f172a"
    muted = "#9ca3af" if is_dark else "#475569"
    border = "rgba(148, 163, 184, 0.28)" if is_dark else "rgba(49, 51, 63, 0.14)"
    button_bg = "linear-gradient(180deg, rgba(30,41,59,0.98), rgba(15,23,42,0.98))" if is_dark else "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98))"
    note_bg = "rgba(59, 130, 246, 0.13)" if is_dark else "rgba(37, 99, 235, 0.08)"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}
        h1, h2, h3, h4, h5, h6, p, li, label, span {{
            color: {text};
        }}
        [data-testid="stSidebar"] {{
            background: {card};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] * {{
            color: {text};
        }}
        [data-testid="stMetric"], [data-testid="stDataFrame"], .stAlert {{
            border-radius: 16px;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            border-radius: 14px;
            padding: 0.72rem 0.85rem;
            border: 1px solid {border};
            background: {button_bg};
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
            text-align: left;
            font-weight: 650;
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-1px);
            border-color: rgba(37, 99, 235, 0.62);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
        }}
        .nav-current {{
            border-radius: 14px;
            padding: 0.78rem 0.9rem;
            margin: 0.18rem 0 0.45rem 0;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white !important;
            font-weight: 750;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.26);
        }}
        .nav-current * {{ color: white !important; }}
        .nav-caption {{
            color: rgba(255,255,255,0.84) !important;
            font-size: 0.78rem;
            font-weight: 450;
            margin-top: 0.1rem;
        }}
        .theme-card {{
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            margin: 0.7rem 0 1rem 0;
            background: {soft_card};
            border: 1px solid {border};
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        }}
        .theme-title {{
            font-weight: 800;
            margin-bottom: 0.15rem;
        }}
        .theme-caption {{
            color: {muted} !important;
            font-size: 0.78rem;
        }}
        .bubble-note {{
            border-radius: 16px;
            padding: 0.8rem 1rem;
            background: {note_bg};
            border: 1px solid rgba(37, 99, 235, 0.22);
        }}
        .workflow-card {{
            border-radius: 22px;
            padding: 1.2rem;
            background: {card};
            border: 1px solid {border};
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
        }}
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
    st.sidebar.title("📦 Dashboard")
    st.sidebar.caption("Forecast-driven inventory replenishment")

    theme_label = "Dark mode" if st.session_state.get("dark_mode", False) else "Light mode"
    st.sidebar.markdown(
        f'<div class="theme-card"><div class="theme-title">🌓 Theme</div><div class="theme-caption">Current: {theme_label}</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.toggle("🌙 Enable dark mode", key="dark_mode")
    st.sidebar.markdown("---")

    for page_name, icon, caption in NAV_ITEMS:
        if current_page == page_name:
            st.sidebar.markdown(
                f'<div class="nav-current">{icon} {page_name}<div class="nav-caption">{caption}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            if st.sidebar.button(f"{icon}  {page_name}", key=f"nav_{page_name}"):
                set_page(page_name)

    st.sidebar.markdown("---")
    if current_page == CHATBOT_PAGE:
        st.sidebar.markdown(
            '<div class="nav-current">💬 Research Chatbot<div class="nav-caption">Ask about data, models, and results</div></div>',
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button("💬  Research Chatbot", key="nav_chatbot"):
            set_page(CHATBOT_PAGE)

    st.sidebar.markdown("---")
    st.sidebar.caption("AI2013 / DAP391m Group 3")


def render_chatbot_bubble():
    """Inject a draggable floating chatbot bubble that appears on every page."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const old = doc.getElementById('dap-chatbot-bubble');
            if (old) { old.remove(); }

            const bubble = doc.createElement('div');
            bubble.id = 'dap-chatbot-bubble';
            bubble.innerHTML = '<div class="dap-bubble-icon">💬</div><div class="dap-bubble-text">Ask<br>Research</div>';
            bubble.style.cssText = `
                position: fixed;
                right: 26px;
                bottom: 26px;
                z-index: 2147483647;
                width: 84px;
                height: 84px;
                border-radius: 50%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 1px;
                cursor: grab;
                user-select: none;
                color: #ffffff;
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
                box-shadow: 0 16px 34px rgba(37, 99, 235, 0.36), 0 4px 10px rgba(15, 23, 42, 0.18);
                border: 2px solid rgba(255,255,255,0.88);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            `;

            const style = doc.createElement('style');
            style.id = 'dap-chatbot-bubble-style';
            style.innerHTML = `
                #dap-chatbot-bubble:hover { transform: scale(1.05); box-shadow: 0 20px 44px rgba(37, 99, 235, 0.46), 0 8px 18px rgba(15,23,42,0.22); }
                #dap-chatbot-bubble .dap-bubble-icon { font-size: 27px; line-height: 1; }
                #dap-chatbot-bubble .dap-bubble-text { font-size: 11px; line-height: 1.05; font-weight: 750; text-align: center; letter-spacing: .01em; }
                @media (max-width: 700px) { #dap-chatbot-bubble { width: 72px; height: 72px; right: 18px; bottom: 18px; } }
            `;
            const oldStyle = doc.getElementById('dap-chatbot-bubble-style');
            if (oldStyle) { oldStyle.remove(); }
            doc.head.appendChild(style);
            doc.body.appendChild(bubble);

            let isDragging = false;
            let moved = false;
            let offsetX = 0;
            let offsetY = 0;

            function startDrag(clientX, clientY) {
                const rect = bubble.getBoundingClientRect();
                offsetX = clientX - rect.left;
                offsetY = clientY - rect.top;
                isDragging = true;
                moved = false;
                bubble.style.cursor = 'grabbing';
            }

            function dragTo(clientX, clientY) {
                if (!isDragging) return;
                moved = true;
                const maxX = doc.documentElement.clientWidth - bubble.offsetWidth - 8;
                const maxY = doc.documentElement.clientHeight - bubble.offsetHeight - 8;
                let x = Math.min(Math.max(8, clientX - offsetX), maxX);
                let y = Math.min(Math.max(8, clientY - offsetY), maxY);
                bubble.style.left = x + 'px';
                bubble.style.top = y + 'px';
                bubble.style.right = 'auto';
                bubble.style.bottom = 'auto';
            }

            function stopDrag() {
                if (!isDragging) return;
                isDragging = false;
                bubble.style.cursor = 'grab';
            }

            bubble.addEventListener('mousedown', function(e) {
                e.preventDefault();
                startDrag(e.clientX, e.clientY);
            });
            doc.addEventListener('mousemove', function(e) { dragTo(e.clientX, e.clientY); });
            doc.addEventListener('mouseup', stopDrag);

            bubble.addEventListener('touchstart', function(e) {
                const t = e.touches[0];
                startDrag(t.clientX, t.clientY);
            }, {passive: true});
            doc.addEventListener('touchmove', function(e) {
                if (!isDragging) return;
                const t = e.touches[0];
                dragTo(t.clientX, t.clientY);
            }, {passive: true});
            doc.addEventListener('touchend', stopDrag);

            bubble.addEventListener('click', function(e) {
                if (moved) return;
                const url = new URL(window.parent.location.href);
                url.searchParams.set('page', 'Research Chatbot');
                window.parent.location.href = url.toString();
            });
        })();
        </script>
        """,
        height=0,
        width=0,
    )

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
    data = apply_display_filters(data)
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
        st.markdown('<div class="workflow-card">', unsafe_allow_html=True)
        st.image(str(workflow_path), caption="End-to-end research workflow: data preparation, model development, and inventory operations")
        st.markdown('</div>', unsafe_allow_html=True)

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
    st.caption("CatBoost and Moving Average are excluded from the app display. The SNAP feature is also excluded from the feature-importance view.")
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
    st.caption("Simulation focuses on Naive, Fixed Rule, LightGBM, and Tuned LightGBM. XGBoost and Random Forest are excluded from simulation because they are used only for forecasting comparison in this app.")
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


def get_chatbot_response(question, data):
    """Rule-based bilingual research assistant for the dashboard."""
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

def chatbot_page(data):
    st.title("💬 Research Assistant Chatbot")
    st.markdown(
        "Ask questions about the dataset, forecasting models, feature importance, inventory simulation, metrics, and research findings. "
        "Bạn cũng có thể hỏi bằng tiếng Việt; chatbot sẽ tự trả lời bằng tiếng Việt."
    )

    st.markdown(
        """
        <div class="bubble-note">
        <b>Tip:</b> The draggable chat bubble appears on every page. Drag it with your mouse, or click it to open this chatbot page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Example questions: What dataset is used? Which model is best? What is RMSSE? Vì sao XGBoost và Random Forest không dùng trong simulation? Total cost là gì?"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", "Hi! Mình có thể giải thích dashboard này bằng English hoặc tiếng Việt. Ask me about M5 data, forecasting models, inventory simulation, costs, or final conclusions.")
        ]

    c1, c2, c3, c4 = st.columns(4)
    suggested = None
    if c1.button("What dataset is used?"):
        suggested = "What dataset is used?"
    if c2.button("Which model is best?"):
        suggested = "Which model is best?"
    if c3.button("Total cost là gì?"):
        suggested = "Total cost là gì?"
    if c4.button("Model nào tốt nhất?"):
        suggested = "Model nào tốt nhất?"

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    user_question = st.chat_input("Ask a question / Hỏi về nghiên cứu này...")
    if suggested and not user_question:
        user_question = suggested

    if user_question:
        answer = get_chatbot_response(user_question, data)
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

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
    theme_mode = get_theme_mode()
    px.defaults.template = "plotly_dark" if theme_mode == "Dark" else "plotly_white"
    inject_app_style(theme_mode)
    render_chatbot_bubble()
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
    elif page == "Research Chatbot":
        chatbot_page(data)
    elif page == "Final Comparison":
        conclusion_page(data)

if __name__ == "__main__":
    main()
