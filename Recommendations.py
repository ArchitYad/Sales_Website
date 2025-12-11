import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
def show_recommendations():
    # 2_Recommendations.py
    st.set_page_config(page_title="Supermarket Recommendations", layout="wide")
    st.title("🎯 Recommendations & Campaigns")

    # -------------------------
    # Check if Analysis results exist
    # -------------------------
    if 'rfm' not in st.session_state or 'df' not in st.session_state:
        st.warning("Please run the Analysis page first.")
        st.stop()

    df = st.session_state.df
    rfm = st.session_state.rfm
    monetary_col = None
    for col in ["Sales", "Total", "gross income"]:
        if col in df.columns:
            monetary_col = col
            break

    store_attr = st.session_state.get('store_attr', None)
    clf = st.session_state.get('clf', None)
    X = st.session_state.get('X', None)

    # -------------------------
    # 1️⃣ Campaigns by RFM / K-Means segments
    # -------------------------
    st.subheader("1️⃣ Campaigns for Customer Segments")

    def generate_segment_campaigns(rfm_df):
        campaigns = []
        rec_q = rfm_df['Recency'].quantile([0.33, 0.66]).values
        freq_q = rfm_df['Frequency'].quantile([0.33, 0.66]).values
        mon_q = rfm_df['Monetary'].quantile([0.33, 0.66]).values

        for idx, row in rfm_df.iterrows():
            seg_label = f"Customer {row.name}"
            recency, freq, mon = row['Recency'], row['Frequency'], row['Monetary']
            if (recency <= rec_q[0]) and (freq >= freq_q[1]) and (mon >= mon_q[1]):
                campaigns.append({'customer': seg_label, 'segment': 'VIP',
                                  'suggestion': 'Invite to VIP program; offer 15% off on premium products; early access to new offers.'})
            elif (recency > rec_q[1]) and (freq <= freq_q[0]):
                campaigns.append({'customer': seg_label, 'segment': 'Churn-risk',
                                  'suggestion': 'Send win-back coupon (20% off), personalized email with best-sellers and free delivery.'})
            elif (freq >= freq_q[1]) and (mon < mon_q[0]):
                campaigns.append({'customer': seg_label, 'segment': 'Frequent low-spender',
                                  'suggestion': 'Promote bundle upgrades and loyalty points for incremental spend.'})
            else:
                campaigns.append({'customer': seg_label, 'segment': 'Regular',
                                  'suggestion': 'Routine promotional messages; include cross-sell of frequent combos.'})
        return pd.DataFrame(campaigns)

    seg_campaigns_df = generate_segment_campaigns(rfm)
    st.dataframe(seg_campaigns_df.head(10))

    csv = seg_campaigns_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download segment campaigns CSV", data=csv, file_name="segment_campaigns.csv", mime="text/csv")

    # -------------------------
    # 2️⃣ Product Offers & Bundle Suggestions
    # -------------------------
    st.subheader("2️⃣ Product Offers & Bundles")
    bundles_df = pd.DataFrame()
    if {'Gender','Product line',monetary_col}.issubset(df.columns):
        top_by_group = (df.groupby(['Gender','Product line'])[monetary_col]
                        .sum().reset_index()
                        .sort_values(['Gender', monetary_col], ascending=[True,False]))
        bundles = []
        for (g), group_df in top_by_group.groupby('Gender'):
            top_lines = group_df['Product line'].unique()[:2]
            if len(top_lines) >= 2:
                bundles.append({'gender': g,
                                'bundle': f"{top_lines[0]} + {top_lines[1]}",
                                'suggestion': f"Offer 10% off when buying {top_lines[0]} with {top_lines[1]} for {g} customers."})
        if bundles:
            bundles_df = pd.DataFrame(bundles)
            st.table(bundles_df)
    else:
        st.info("Not enough columns to compute bundles (require Customer type, Gender, Product line).")

    # -------------------------
    # 3️⃣ Decision Tree insights
    # -------------------------
    st.subheader("3️⃣ Decision Tree Targeting Insights")
    if clf is not None and X is not None:
        importances = clf.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(5)
        st.bar_chart(feat_imp)
        st.write("Suggested action: Target promotions based on these features.")
    else:
        st.info("Decision Tree model not found; ensure Analysis page trained it.")

    # -------------------------
    # 4️⃣ Store Performance Actions
    # -------------------------
    st.subheader("4️⃣ Store Performance & Actions")
    actions_df = pd.DataFrame()
    if store_attr is not None:
        sa = store_attr.copy()
        sa['sales_per_branch'] = sa['Attractiveness']
        median_sales = sa['sales_per_branch'].median()
        actions = []
        for _, row in sa.iterrows():
            if row['sales_per_branch'] < median_sales:
                actions.append({'branch': row['Branch'], 'action': 'Underperforming: run local promotions, adjust stock for high-margin items.'})
            else:
                actions.append({'branch': row['Branch'], 'action': 'Performing well: maintain inventory, test premium product launches and loyalty incentives.'})
        if actions:
            actions_df = pd.DataFrame(actions)
            st.table(actions_df)
    else:
        st.info("Store attribute data not available.")

    # -------------------------
    # 5️⃣ New Store Location Recommendation
    # -------------------------
    st.subheader("5️⃣ New Store Location Recommendation")
    
    existing_cities = ["Mandalay State", "Yangon Division",  "Nay Pyi Taw State"]
    
    # === Step 2: File group mapping ===
    file_groups = {
        "Group1": ("table_1.xlsx", ["Union", "Kachin State", "Kayah State", "Kayin State"]),
        "Group2": ("table_2.xlsx", ["Chin State", "Sagaing Division", "Tanintharyi Division", "Bago Division"]),
        "Group3": ("table_3.xlsx", ["Magway Division", "Mandalay State", "Mon State", "Rakhine State"]),
        "Group4": ("table_4.xlsx", ["Yangon Division", "Shan State", "Ayeyarwady State", "Nay Pyi Taw State"])
    }
    
    # Food & non-food subcategories
    food_items = [
        "Rice","Pulses","Cooking oil and fats","Meat","Eggs","Fish and crustacea (fresh)","Vegetables",
        "Fruits","Fish and crustacea (dried)","Wheat and Rice products","Food Taken Outside Home",
        "Ngapi and nganpyaye","Spices and condiments","Beverages","Sugar and other food","Milk and milk products"
    ]
    
    non_food_items = [
        "Tobacco","Fuel and light","Travelling expenses (Local)","Travelling expenses (Journey)",
        "Clothing and apparel","Personal use goods","Cleansing and toilet","Crockery","Furniture",
        "House rent and repairs","Education","Stationery and school supplies","Medical care",
        "Recreation","Charity and ceremonials","Other expenses","Other household goods"
    ]
    
    # === Step 3: Load and combine data ===
    region_totals = {}
    for group_name, (url, regions) in file_groups.items():
        df = pd.read_excel(url, header=1)
        df = df.rename(columns={df.columns[1]: "Particulars"})
        
        for idx, region in enumerate(regions):
            value_col = "Value" if idx == 0 else f"Value.{idx}"
            region_data = df[["Particulars", value_col]].copy()
            region_data = region_data.rename(columns={value_col: "Value"})
            region_data = region_data.set_index("Particulars")
            region_totals[region] = region_data
    
    # === Step 4: Remove existing retail cities ===
    available_regions = [r for r in region_totals.keys() if r not in existing_cities]
    
    # === Step 5: Compute top 3 regions by total expenditure ===
    top3_regions = sorted(
        available_regions,
        key=lambda x: region_totals[x].loc["HOUSEHOLD EXPENDITURE TOTAL", "Value"],
        reverse=True
    )[:3]
    
    # === Step 6: Map coordinates (OpenStreetMap) ===
    region_coords = {
        "Union": [21.9162, 95.9560],
        "Kachin State": [25.57, 97.33],
        "Kayah State": [19.33, 96.58],
        "Kayin State": [16.73, 97.60],
        "Chin State": [21.95, 93.73],
        "Sagaing Division": [22.00, 95.00],
        "Tanintharyi Division": [12.25, 99.50],
        "Bago Division": [17.33, 96.50],
        "Magway Division": [20.15, 95.55],
        "Mon State": [16.55, 97.73],
        "Rakhine State": [19.90, 94.85],
        "Shan State": [21.90, 97.80],
        "Ayeyarwady State": [16.77, 94.73]
    }
    
    # === Step 7: Display recommendations ===
    st.subheader("5️⃣ New Store Location Recommendation")
    m = folium.Map(location=[21.9162, 95.9560], zoom_start=6, tiles="OpenStreetMap")
    
    for region in top3_regions:
        data = region_totals[region]
        food_total = data.loc["FOOD AND BEVERAGES TOTAL", "Value"]
        non_food_total = data.loc["NON-FOOD TOTAL", "Value"]
        total = data.loc["HOUSEHOLD EXPENDITURE TOTAL", "Value"]
    
        food_share = food_total / total * 100
        non_food_share = non_food_total / total * 100
        focus = "🛒 Food & FMCG" if food_share > non_food_share else "💡 Non-Food Retail"
    
        # Top subcategories
        top_food = data.loc[food_items, "Value"].sort_values(ascending=False).head(3).index.tolist()
        top_non_food = data.loc[non_food_items, "Value"].sort_values(ascending=False).head(3).index.tolist()
    
        st.write(f"✅ **{region}** — {focus}")
        st.write(f"   🍚 Top Food Items: {', '.join(top_food)}")
        st.write(f"   🧾 Top Non-Food Items: {', '.join(top_non_food)}")
    
        coords = region_coords.get(region)
        if coords:
            folium.Marker(
                location=coords,
                popup=f"{region}\n{focus}",
                tooltip=region
            ).add_to(m)
    
    # Display Folium map
    st_folium(m, width=700, height=500)

    # -------------------------
    # 6️⃣ AI Chat – Strategic Recommendation Engine (RAG + LLM)
    # -------------------------
    st.markdown("---")
    st.subheader("🤖 AI Strategy Assistant (RAG + LLM)")
    
    # Create a sidebar floating chat window
    with st.sidebar:
        st.title("📌 Retail AI Assistant")
        st.caption("LLM + RAG + Market Intelligence")
    
        # Load RAG index only once
        from rag_utils import load_rag_index, get_llm_chain
    
        if "rag_index" not in st.session_state:
            with st.spinner("Building knowledge base from government tables…"):
                st.session_state.rag_index = load_rag_index()
    
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
        qa = get_llm_chain(st.session_state.rag_index)
    
        # Chat input
        user_msg = st.text_input(
            "Ask something…",
            placeholder="e.g., 'Where should our next store open?'"
        )
    
        # Handle response
        if user_msg:
            with st.spinner("Thinking…"):
                llm_reply = qa.run(user_msg)
    
            # Store in chat memory
            st.session_state.chat_history.append(("user", user_msg))
            st.session_state.chat_history.append(("bot", llm_reply))
    
        # Display chat
        st.write("### 💬 Chat History")
        for speaker, msg in st.session_state.chat_history:
            if speaker == "user":
                st.markdown(f"🧑 **You:** {msg}")
            else:
                st.markdown(f"🤖 **AI:** {msg}")
    
        # Optional: clear chat
        if st.button("Clear Chat"):
            st.session_state.chat_history = []


    # -------------------------
    # 7 Download all recommendations
    # -------------------------
    all_reco = {
        'segment_campaigns': seg_campaigns_df,
        'bundles': bundles_df,
        'store_actions': actions_df
    }

    def combine_recos_to_csv(dict_of_dfs):
        out = []
        for k, v in dict_of_dfs.items():
            if v.shape[0] > 0:
                section_col = pd.DataFrame({'section': [k]*v.shape[0]})
                out.append(pd.concat([section_col, v.reset_index(drop=True)], axis=1))
        if out:
            combined = pd.concat(out, axis=0)
            return combined.to_csv(index=False).encode('utf-8')
        return None

    csv_all = combine_recos_to_csv(all_reco)
    if csv_all:
        st.download_button("Download All Recommendations CSV", data=csv_all, file_name="recommendations.csv", mime="text/csv")
    else:
        st.info("No recommendations to export yet.")
