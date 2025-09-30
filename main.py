import streamlit as st
from Home import show_home
from Analysis import show_analysis
from Recommendations import show_recommendations

st.set_page_config(page_title="Supermarket Dashboard", layout="wide")
st.title("🛒 Supermarket Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["Home", "Analysis", "Recommendations"])

with tab1:
    show_home()  # Your Home.py content wrapped in a function

with tab2:
    show_analysis()  # Analysis.py content

with tab3:
    show_recommendations()  # Recommendations.py content
