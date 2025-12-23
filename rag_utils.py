# rag_utils.py
import os
import pandas as pd
from typing import Dict, List, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq


# -----------------------------
# Load government tables
# -----------------------------
def _load_tables(file_groups: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    region_totals = {}
    for _, (file, regions) in file_groups.items():
        df = pd.read_excel(file, header=1)
        df = df.rename(columns={df.columns[1]: "Particulars"})
        for idx, region in enumerate(regions):
            value_col = "Value" if idx == 0 else f"Value.{idx}"
            region_df = df[["Particulars", value_col]].copy()
            region_df = region_df.rename(columns={value_col: "Value"})
            region_df.set_index("Particulars", inplace=True)
            region_totals[region] = region_df
    return region_totals


# -----------------------------
# Build RAG documents
# -----------------------------
def _build_docs(region_totals: Dict[str, pd.DataFrame]) -> List[str]:
    docs = []
    for region, df in region_totals.items():
        total = df.loc["HOUSEHOLD EXPENDITURE TOTAL"]["Value"] if "HOUSEHOLD EXPENDITURE TOTAL" in df.index else "unknown"
        docs.append(f"{region}: Total household expenditure is {total}.")
        for idx, row in df.iterrows():
            val = row.get("Value", None)
            if val is not None:
                docs.append(f"{region} | {idx} | Expenditure: {val}")
    return docs


# -----------------------------
# Build Vector DB
# -----------------------------
def load_rag_index(file_groups: Dict = None):
    if file_groups is None:
        file_groups = {
            "Group1": ("table_1.xlsx", ["Union", "Kachin State", "Kayah State", "Kayin State"]),
            "Group2": ("table_2.xlsx", ["Chin State", "Sagaing Division", "Tanintharyi Division", "Bago Division"]),
            "Group3": ("table_3.xlsx", ["Magway Division", "Mandalay State", "Mon State", "Rakhine State"]),
            "Group4": ("table_4.xlsx", ["Yangon Division", "Shan State", "Ayeyarwady State", "Nay Pyi Taw State"])
        }

    region_totals = _load_tables(file_groups)
    docs = _build_docs(region_totals)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(docs, embedding=embeddings)
    return vectordb


# -----------------------------
# LLM + RAG Chain with Summary
# -----------------------------
def get_llm_chain(vectordb, extra_context: str = ""):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
You are a retail strategy AI assistant.

Use ONLY the information below:
- Myanmar Government household expenditure data
- Live market and social signals (if provided)

GOVERNMENT CONTEXT:
{{context}}

LIVE SIGNALS:
{extra_context}

QUESTION:
{{question}}

Provide a concise strategic recommendation in 4–5 bullet points, then summarize into a short paragraph.
"""
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def groq_llm(prompt_text) -> str:
        # Ensure the prompt is always a string
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        resp = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.4
        )

        full_output = resp.choices[0].message["content"]
        # Take first 6 lines as a safe summarization for UI
        summary = "\n".join(full_output.split("\n")[:6])
        return summary

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | groq_llm
        | StrOutputParser()
    )

    return chain
