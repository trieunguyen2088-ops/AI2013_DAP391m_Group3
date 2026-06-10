# M5 Forecast-Driven Inventory Replenishment Dashboard

This Streamlit app is a research demo for **Inventory Replenishment Optimization Using Sales Signals**.

The app does not load the full Walmart M5 raw dataset. Instead, it uses small processed output files generated from the project notebooks:

- forecasting metrics
- forecast time-series samples
- inventory simulation summaries
- inventory time-series samples
- LightGBM tuning and feature-importance outputs

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
