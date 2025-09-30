import streamlit as st
from Home import show_home
from Analysis import show_analysis
from Recommendations import show_recommendations

st.set_page_config(page_title="Supermarket Dashboard", layout="wide")
st.title("🛒 Supermarket Dashboard")

# Tabs
tab = st.sidebar.radio("Navigate", ["Home", "Analysis", "Recommendations"])

if tab == "Home":
    show_home()  # Your Home.py content wrapped in a function

elif tab == "Analysis":
    show_analysis()  # Analysis.py content

elif tab == "Recommendations":
    show_recommendations()  # Recommendations.py content
