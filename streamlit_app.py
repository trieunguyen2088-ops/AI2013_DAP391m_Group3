import streamlit as st

st.set_page_config(
    page_title="M5 Inventory Replenishment Dashboard",
    layout="wide"
)

st.title("M5 Inventory Replenishment Dashboard")

st.markdown("""
This app demonstrates how demand forecasting models can be used to support
inventory replenishment decisions using Walmart M5 sales data.
""")

st.header("Project Pipeline")

st.markdown("""
**M5 Sales Data**  
→ **Feature Engineering**  
→ **Forecasting Models**  
→ **Inventory Simulation**  
→ **Cost Comparison**
""")

st.success("The Streamlit app is running successfully.")
