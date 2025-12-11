import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def load_rag_index():
    """Load table_1 to table_4 and build a FAISS index."""
    files = ["table_1.xlsx", "table_2.xlsx", "table_3.xlsx", "table_4.xlsx"]
    dfs = [pd.read_excel(f) for f in files]

    docs = []
    for df in dfs:
        df = df.fillna("")
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb

def get_llm_chain(vectordb):
    from langchain.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key="YOUR_GROQ_API_KEY"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question", "trends", "market_news"],
        template="""
        You are an **AI Retail Strategy Assistant**.

        Use the following sources:
        1. **Government expenditure tables (RAG Context)** – customer spending behavior by region.
        2. **Social Media Trends** – extracted from Twitter/Reddit (focus on product popularity & sentiment).
        3. **Market News** – to detect price rise/fall, shortages, demand surges.
        
        Your tasks:
        - Recommend product mix for the region.
        - Suggest discount levels.
        - Identify possible new store locations.
        - Forecast product demand based on spending & current trends.
        - Give short, actionable insights.

        === GOVERNMENT DATA CONTEXT (RAG) ===
        {context}

        === SOCIAL TRENDS (Twitter/Reddit) ===
        {trends}

        === MARKET NEWS SUMMARY ===
        {market_news}

        === QUESTION ===
        {question}

        Provide **5 crisp bullet points**. Be precise and data-driven.
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )
