# M5 Forecast-Driven Inventory Replenishment Dashboard

This Streamlit app presents the processed outputs of the AI2013 / DAP391m Group 3 research project.

## Main features

- Forecasting performance dashboard
- Actual vs forecast visualization
- Inventory simulation results
- Inventory time-series explorer
- What-if replenishment calculator
- Final forecasting vs inventory trade-off comparison
- Floating Research Assistant chatbot powered by Gemini API when a key is configured

## Deploy on Streamlit Cloud

Use:

```text
Main file path: streamlit_app.py
```

## Gemini API setup

The chatbot will use Gemini if you add a secret named:

```text
GEMINI_API_KEY
```

In Streamlit Community Cloud, open the app settings, go to **Secrets**, and add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

If no key is configured, the chatbot still works using local fallback answers.

## Data note

The app does not load the full raw Walmart M5 dataset. It uses small processed output files in the `data/` folder for fast deployment.
