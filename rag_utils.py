# rag_utils.py

import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# External Signals
import requests
import praw   # Reddit API wrapper


# ===============================================================
# 1. LOAD GOVERNMENT XLSX TABLES INTO RAG VECTOR STORE
# ===============================================================

def load_rag_index(data_folder="gov_tables"):
    """
    Loads table_1.xlsx to table_4.xlsx and builds a FAISS vectorstore.
    """

    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = []

    for file in ["table_1.xlsx", "table_2.xlsx", "table_3.xlsx", "table_4.xlsx"]:
        path = os.path.join(data_folder, file)
        if not os.path.exists(path):
            continue

        df = pd.read_excel(path)
        df = df.fillna("")

        for _, row in df.iterrows():
            txt = " | ".join([f"{c}: {row[c]}" for c in df.columns])
            docs.append(txt)

    vectordb = FAISS.from_texts(docs, embeds)
    return vectordb


# ===============================================================
# 2. FETCH NEWS (Market Trends, Inflation, Retail)
# ===============================================================

def fetch_market_news(query="retail market Myanmar"):
    try:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&sortBy=publishedAt&pageSize=4&apiKey={os.getenv('NEWS_API_KEY')}"
        )
        r = requests.get(url).json()
        articles = r.get("articles", [])
        cleaned = []

        for a in articles:
            cleaned.append(f"{a['title']} — {a.get('description','')}")
        return cleaned
    except:
        return ["NewsAPI unavailable or invalid key."]


# ===============================================================
# 3. FETCH SOCIAL MEDIA TRENDS (Twitter/X or Reddit)
# ===============================================================

def fetch_reddit_trends(subreddit="economy", limit=3):
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="retail-trend-bot"
        )

        posts = reddit.subreddit(subreddit).hot(limit=limit)
        out = []
        for p in posts:
            out.append(f"{p.title} — {p.selftext[:200]}")
        return out
    except:
        return ["Reddit API unavailable."]


def fetch_twitter_trends(keyword="Myanmar retail"):
    """
    You can replace with official Twitter API v2 bearer token.
    """
    try:
        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        headers = {"Authorization": f"Bearer {bearer}"}
        url = (
            f"https://api.twitter.com/2/tweets/search/recent?"
            f"query={keyword}&max_results=5"
        )
        r = requests.get(url, headers=headers).json()
        data = r.get("data", [])
        return [d["text"] for d in data]
    except:
        return ["Twitter API unavailable or invalid key."]


# ===============================================================
# 4. BUILD THE LLM CHAIN (RAG + NEWS + SOCIAL SIGNALS)
# ===============================================================

def get_llm_chain(vectordb):

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["gov_context", "news_context", "social_context", "question"],
        template="""
You are an AI retail strategist. 
Combine:
1. **Government expenditure tables** (RAG)  
2. **Market news signals**  
3. **Social sentiment (Reddit/Twitter)**  

Provide recommendations using ONLY these factual inputs.

====================  
📘 GOVERNMENT DATA  
{gov_context}
====================  

📰 MARKET NEWS  
{news_context}

====================  
💬 SOCIAL SENTIMENT  
{social_context}

====================  

🎯 QUESTION:  
{question}

-----------------------  
Provide an actionable answer in **4–6 bullet points**.  
Avoid hallucinations.
"""
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    def ask(question):
        # 1. RAG retrieval
        gov_context = "\n".join(retriever.get_relevant_documents(question)[i].page_content
                                for i in range(4))

        # 2. Market news
        news_list = fetch_market_news(query="Myanmar retail economy")
        news_context = "\n".join(news_list)

        # 3. Social signals
        reddit_context = "\n".join(fetch_reddit_trends("retail"))
        twitter_context = "\n".join(fetch_twitter_trends("Myanmar retail"))
        social_context = reddit_context + "\n" + twitter_context

        # 4. Run LLM
        return llm.predict(
            prompt.format(
                gov_context=gov_context,
                news_context=news_context,
                social_context=social_context,
                question=question
            )
        )

    return ask
