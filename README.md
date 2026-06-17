# M5 Forecast-Driven Inventory Replenishment Dashboard

Streamlit dashboard for AI2013 / DAP391m Group 3.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud

- Repository: `trieunguyen2088-ops/AI2013_DAP391m_Group3`
- Branch: `main`
- Main file path: `streamlit_app.py`

## Gemini API key

The chatbot can use Gemini API if a key is configured. In Streamlit Cloud, open **App settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

If no key is configured, the chatbot automatically uses local fallback answers.

## Version notes

- Chatbot is a non-modal fixed panel in the bottom-right corner.
- The dashboard remains scrollable while the chatbot is open.
- The chatbot panel has a solid theme-aware background.
- The app follows Streamlit native System / Light / Dark theme.
