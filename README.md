# M5 Forecast-Driven Inventory Replenishment Dashboard

This Streamlit app is a research demo for **Inventory Replenishment Optimization Using Sales Signals**.

The app does not load the full Walmart M5 raw dataset. Instead, it uses small processed output files generated from the project notebooks:

- forecasting metrics
- forecast time-series samples
- inventory simulation summaries
- inventory time-series samples
- LightGBM tuning and feature-importance outputs
- a bilingual rule-based research assistant chatbot for questions about the data, models, metrics, and simulation
- a draggable floating chatbot bubble that appears on every dashboard page
- professional button-based navigation instead of a simple sidebar radio

## How to run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud settings

Use these values when deploying from GitHub:

```text
Repository: trieunguyen2088-ops/AI2013_DAP391m_Group3
Branch: main
Main file path: streamlit_app.py
App URL: m5-inventory-simulator
```

## Suggested repository structure

```text
AI2013_DAP391m_Group3/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── data/
└── assets/
```


## Main app pages

- Overview
- Forecasting Performance
- Actual vs Forecast
- Inventory Simulation Results
- Inventory Time-series Explorer
- What-if Simulator
- Research Chatbot
- Final Comparison

## Research Chatbot

The chatbot is a lightweight rule-based assistant. It does not require OpenAI API keys or any external service. It answers common questions about the Walmart M5 data, forecasting models, RMSSE metrics, feature importance, lead time, inventory cost, and the main research conclusions. It can detect Vietnamese questions and respond in Vietnamese. A draggable floating chat bubble is available on every page to open the chatbot quickly.

## Version v5 update

This version adds a light/dark theme toggle in the sidebar. The selected theme changes the main dashboard background, sidebar, navigation buttons, cards, and Plotly chart template. The Overview page also displays the updated research workflow image.


## Version v7 updates

- Research Chatbot is hidden from the sidebar navigation and opened through the floating chat bubble.
- The floating bubble now uses a more reliable click action while remaining draggable.
- Added stronger Plotly light/dark styling so chart labels, text, legends, and backgrounds follow the selected theme.
- Chat history can still be cleared inside the chatbot page.
