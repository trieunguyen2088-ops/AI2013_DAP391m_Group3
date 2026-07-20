# DAP391m Streamlit App

Forecast-based segment-aware replenishment simulation dashboard for the M5 retail sales project.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app uses saved outputs from notebooks `01 -> 02 -> 03 -> 04`. It does not retrain models during demo. The recommendation page uses saved forecasts and validation safety-stock profiles to compute reorder point, target inventory, and suggested order quantity.

Optional saved model inference is available for LightGBM, LightGBM Tuned, and XGBoost when the model files and processed feature table are available in the parent project folder.
