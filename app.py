import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Supermarket Dashboard", layout="wide")
st.title("🛒 Supermarket Sales Dashboard")

# -------------------------
# File Upload (Safe Logic)
# -------------------------
uploaded_file = st.file_uploader("📂 Upload Supermarket Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Store in session memory only (not on disk)
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv(uploaded_file)

    df = st.session_state.data

    # -------------------------
    # Data Preprocessing
    # -------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df["InvoiceDate"] = df["Date"] + pd.to_timedelta(df["Time"])
    df["CustomerID"] = df["Invoice ID"].astype(str)

    # -------------------------
    # Objective 1: Customer Profiling (SEQ-RFM)
    # -------------------------
    st.header("1️⃣ Customer Profiling (SEQ-RFM)")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice ID": "count",
        "Total": "sum"
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Add sequence of product lines per customer
    seq = df.groupby("CustomerID")["Product line"].apply(lambda x: " → ".join(x))
    rfm = rfm.join(seq)

    st.write(rfm.head())

    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency",
                         hover_data=["Product line"], title="RFM Scatter Plot")
    st.plotly_chart(fig_rfm, use_container_width=True)

    # -------------------------
    # Objective 2: Shopping Behavior (Apriori + Decision Tree)
    # -------------------------
    st.header("2️⃣ Shopping Behavior (Apriori + Decision Tree)")

    # Market Basket Analysis
    basket = pd.crosstab(df["Invoice ID"], df["Product line"])
    frequent_items = apriori(basket, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    st.write("Frequent Itemsets:", frequent_items.head())
    st.write("Association Rules:", rules.head())

    # Peak shopping times
    df["Hour"] = df["Time"].apply(lambda x: int(x.split(":")[0]))
    peak_sales = df.groupby("Hour")["Total"].sum().reset_index()
    fig_peak = px.line(peak_sales, x="Hour", y="Total", title="Sales by Hour")
    st.plotly_chart(fig_peak, use_container_width=True)

    # -------------------------
    # Objective 3: Location Analysis (Dynamic Huff Model)
    # -------------------------
    st.header("3️⃣ Location Analysis (Dynamic Huff Model)")

    # Store attractiveness (sales by branch)
    store_attr = df.groupby("Branch")["Total"].sum().reset_index()
    store_attr.columns = ["Branch", "Attractiveness"]

    # Simulate distance matrix (since dataset doesn’t have lat/lon)
    branches = store_attr["Branch"].unique()
    customers = df["CustomerID"].unique()

    np.random.seed(42)
    distances = pd.DataFrame(
        np.random.randint(1, 20, size=(len(customers), len(branches))),
        index=customers, columns=branches
    )

    alpha, beta = 1, 1
    huff = pd.DataFrame(index=customers, columns=branches)

    for b in branches:
        huff[b] = (store_attr.set_index("Branch").loc[b, "Attractiveness"] ** alpha) / (distances[b] ** beta)

    huff = huff.div(huff.sum(axis=1), axis=0)
    st.write("Huff Model Probabilities (sample):", huff.head())

    # Folium map with branch markers (dummy coords per city)
    city_coords = {"Yangon": [16.8409, 96.1735],
                   "Mandalay": [21.9588, 96.0891],
                   "Naypyitaw": [19.7633, 96.0785]}

    m = folium.Map(location=[20, 96], zoom_start=5)
    for city, coords in city_coords.items():
        total_sales = df[df["City"] == city]["Total"].sum()
        folium.CircleMarker(
            location=coords, radius=total_sales/5000,
            popup=f"{city}: {total_sales:.2f}", color="blue", fill=True
        ).add_to(m)

    st_folium(m, width=700)

    # -------------------------
    # Objective 4: Customer Segmentation (K-Means)
    # -------------------------
    st.header("4️⃣ Customer Segmentation (K-Means)")

    kmeans_data = rfm[["Recency", "Frequency", "Monetary"]]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(kmeans_data)
    rfm["Cluster"] = kmeans.labels_

    fig_seg = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
                            color="Cluster", title="Customer Segmentation (K-Means)")
    st.plotly_chart(fig_seg, use_container_width=True)

else:
    st.info("📥 Please upload the supermarket dataset to begin analysis.")
