def show_home():
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
    # File Upload (CSV or Excel)
    # -------------------------
    uploaded_file = st.file_uploader(
        "📂 Upload Supermarket Dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Auto-detect file type by extension
            file_name = uploaded_file.name.lower()
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
                return

            # Store in session state for access in other pages
            st.session_state.df = df
            st.success("✅ Dataset uploaded successfully!")

            # Display preview
            st.write("### Dataset Sample:")
            st.dataframe(df.head())

            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")

    else:
        st.info("📥 Please upload a CSV or Excel file to continue.")
