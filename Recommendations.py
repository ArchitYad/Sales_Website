import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from rag_utils import load_rag_index, get_llm_chain
from live_sources import fetch_news_summary, fetch_social_summary

def show_recommendations():
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

    st.download_button("Download segment campaigns CSV",
                       data=seg_campaigns_df.to_csv(index=False).encode('utf-8'),
                       file_name="segment_campaigns.csv",
                       mime="text/csv")

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
        for g, group_df in top_by_group.groupby('Gender'):
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
    # Load tables, compute top3 regions etc. (your previous code)
    # ... keep your existing code here for region_totals, top3_regions, and map plotting ...

    # -------------------------
    # 6️⃣ AI Strategy Assistant (LLM + RAG + Live Trends)
    # -------------------------
    st.subheader("🤖 AI Strategy Assistant (LLM + RAG + Live Trends)")
    if "rag_index" not in st.session_state:
        with st.spinner("Building RAG knowledge base..."):
            st.session_state.rag_index = load_rag_index()

    news_summary = fetch_news_summary()
    social_summary = fetch_social_summary()

    extra_context = f"""
Market News:
{news_summary}

Social Trends:
{social_summary}
"""

    qa_chain = get_llm_chain(st.session_state.rag_index, extra_context)

    with st.chat_message("assistant"):
        st.markdown("👋 Ask me about store expansion, discounts, product mix, or regional demand.")

    user_input = st.chat_input("Ask a strategy question...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing with AI..."):
                response = qa_chain.invoke(user_input)
            st.write(response)

    # -------------------------
    # 7️⃣ Download all recommendations
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
