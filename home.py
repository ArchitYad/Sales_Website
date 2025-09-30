# Home.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Supermarket Dashboard Home", layout="wide")
st.title("🛒 Supermarket Sales Dashboard - Home")

st.markdown("""
Welcome! Upload your **Supermarket Sales Dataset** to begin analysis.
This dashboard will provide:
- Customer Profiling (SEQ-RFM)
- Shopping Behavior
- Store Location Analysis
- Customer Segmentation
- Recommendations & Campaigns
""")

# -------------------------
# File Upload (Safe Logic)
# -------------------------
uploaded_file = st.file_uploader("📂 Upload Supermarket Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Store in session state for access in other pages
        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")

        # Show basic info
        st.write("Dataset Sample:")
        st.dataframe(df.head())
        st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

        # Proceed to Analysis button
        if st.button("➡️ Go to Analysis"):
            st.query_params(page="analysis")  # Optional for page navigation
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("📥 Please upload a CSV file to continue.")
