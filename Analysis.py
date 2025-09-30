def show_analysis():
    # 1_Analysis.py
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier
    
    st.set_page_config(page_title="Supermarket Analysis", layout="wide")
    st.title("📊 Supermarket Analysis")
    
    # -------------------------
    # Load Dataset from Home page
    # -------------------------
    if "df" not in st.session_state:
        st.warning("Please upload a dataset from the Home page first.")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # -------------------------
    # Data Preprocessing
    # -------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df["InvoiceDate"] = df["Date"] + pd.to_timedelta(df["Time"])
    df["CustomerID"] = df["Invoice ID"].astype(str)
    
    # Detect monetary column
    monetary_col = None
    for col in ["Sales", "Total", "gross income"]:
        if col in df.columns:
            monetary_col = col
            break
    if monetary_col is None:
        st.error("Dataset missing Sales/Total/gross income column!")
        st.stop()
    
    # -------------------------
    # 1️⃣ Customer Profiling (SEQ-RFM)
    # -------------------------
    st.header("1️⃣ Customer Profiling (SEQ-RFM)")
    
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice ID": "count",
        monetary_col: "sum"
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    # Add product sequence per customer
    if "Product line" in df.columns:
        seq = df.groupby("CustomerID")["Product line"].apply(lambda x: " → ".join(x))
        rfm = rfm.join(seq)
    
    st.write(rfm.head())
    
    fig_rfm = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency",
                         hover_data=rfm.columns, title="RFM Scatter Plot")
    st.plotly_chart(fig_rfm, use_container_width=True)
    
    # -------------------------
    # 2️⃣ Shopping Behavior
    # Peak hour, day, and decision tree
    # -------------------------
    st.header("2️⃣ Shopping Behavior & Market Insights")
    
    # Peak sales by hour
    df["Hour"] = df["Time"].apply(lambda x: int(x.split(":")[0]))
    peak_sales = df.groupby("Hour")[monetary_col].sum().reset_index()
    fig_peak = px.line(peak_sales, x="Hour", y=monetary_col, title="Sales by Hour")
    st.plotly_chart(fig_peak, use_container_width=True)
    
    # Peak day of week
    df["DayOfWeek"] = df["Date"].dt.day_name()
    peak_day = df.groupby("DayOfWeek")[monetary_col].sum().sort_values(ascending=False)
    st.bar_chart(peak_day, use_container_width=True)
    
    # Decision Tree for product_line prediction
    st.subheader("Decision Tree Insights")
    if {'Customer type','Gender','Product line','Unit price','Quantity'}.issubset(df.columns):
        # Simple encoding
        X = pd.get_dummies(df[['Customer type','Gender','Unit price','Quantity']], drop_first=True)
        y = df['Product line']
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)
        st.write("Decision Tree trained to predict Product Line from customer attributes.")
        st.session_state.clf = clf
        st.session_state.X = X
    else:
        st.info("Not enough columns to train Decision Tree.")
    
    # -------------------------
    # 3️⃣ Location Analysis (Dynamic Huff Model)
    # -------------------------
    st.header("3️⃣ Location Analysis (Dynamic Huff Model)")
    
    if "Branch" in df.columns:
        store_attr = df.groupby("Branch")[monetary_col].sum().reset_index()
        store_attr.columns = ["Branch", "Attractiveness"]
        st.write(store_attr)
    
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
        st.write("Huff Probabilities (sample):", huff.head())
        # Folium map with branch markers (dummy coords per city)
            if "City" in df.columns:
                city_coords = {"Yangon": [16.8409, 96.1735],
                               "Mandalay": [21.9588, 96.0891],
                               "Naypyitaw": [19.7633, 96.0785]}
    
                m = folium.Map(location=[20, 96], zoom_start=5)
                for city, coords in city_coords.items():
                    total_sales = df[df["City"] == city][monetary_col].sum()
                    folium.CircleMarker(
                        location=coords, radius=max(3, total_sales/5000),
                        popup=f"{city}: {total_sales:.2f}", color="blue", fill=True
                    ).add_to(m)
    
                st_folium(m, width=700)
        st.session_state.store_attr = store_attr
        st.session_state.huff = huff
    else:
        st.info("Branch information not available for Huff model.")
    
    # -------------------------
    # 4️⃣ Customer Segmentation (K-Means)
    # -------------------------
    st.header("4️⃣ Customer Segmentation (K-Means)")
    
    kmeans_data = rfm[["Recency", "Frequency", "Monetary"]]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(kmeans_data)
    rfm["Cluster"] = kmeans.labels_
    st.session_state.rfm = rfm
    
    fig_seg = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
                            color="Cluster", title="Customer Segmentation (K-Means)")
    st.plotly_chart(fig_seg, use_container_width=True)
    
    # -------------------------
    # Button to go to Recommendations
    # -------------------------
