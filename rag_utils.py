import os
import glob
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# -------------------------
# Build vector DB from XLSX tables
# -------------------------
def load_rag_index(dir_path="government_tables"):
    files = glob.glob(f"{dir_path}/*.xlsx")

    documents = []
    for f in files:
        df = pd.read_excel(f)
        text = df.to_string()
        documents.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = splitter.create_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb


# -------------------------
# LLM + RAG Chain
# -------------------------
def get_llm_chain(vectordb, extra_context=""):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY", "GROQ_API_KEY")
    )

    template = """
You are an AI Retail Strategy Advisor.

Use ONLY:
- Government spending tables (from RAG)
- Live market news & social trends (provided below)

Combine both to offer strategic recommendations.

===== RAG CONTEXT =====
{context}

===== LIVE CONTEXT =====
{extra_context}

===== QUESTION =====
{question}

Reply in 4–6 short bullet points.
"""

    prompt = PromptTemplate(
        input_variables=["context", "question", "extra_context"],
        template=template,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
    )

    chain.combine_documents_chain.llm_chain.prompt = prompt
    chain.extra_context = extra_context

    return chain
