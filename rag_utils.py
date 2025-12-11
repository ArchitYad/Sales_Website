# rag_utils.py

import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter   # <-- FIXED IMPORT
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from groq import Groq

# ----------------------------
# 1. Load Government XLSX Data
# ----------------------------

file_groups = {
    "Group1": ("table_1.xlsx", ["Union", "Kachin State", "Kayah State", "Kayin State"]),
    "Group2": ("table_2.xlsx", ["Chin State", "Sagaing Division", "Tanintharyi Division", "Bago Division"]),
    "Group3": ("table_3.xlsx", ["Magway Division", "Mandalay State", "Mon State", "Rakhine State"]),
    "Group4": ("table_4.xlsx", ["Yangon Division", "Shan State", "Ayeyarwady State", "Nay Pyi Taw State"])
}

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


def load_government_text():
    """
    Convert all XLSX regional expenditure data into consolidated text
    for vector embedding storage.
    """

    region_texts = []

    for group_name, (file, regions) in file_groups.items():
        df = pd.read_excel(file, header=1)
        df = df.rename(columns={df.columns[1]: "Particulars"})

        for idx, region in enumerate(regions):
            value_col = "Value" if idx == 0 else f"Value.{idx}"

            region_data = df[["Particulars", value_col]].copy()
            region_data = region_data.rename(columns={value_col: "Value"})
            region_data = region_data.set_index("Particulars")

            text = f"Region: {region}\n"

            # Total expenditure
            text += f"Total expenditure: {region_data.loc['HOUSEHOLD EXPENDITURE TOTAL', 'Value']}\n"

            # Food vs non-food
            food_total = region_data.loc["FOOD AND BEVERAGES TOTAL", "Value"]
            non_food_total = region_data.loc["NON-FOOD TOTAL", "Value"]
            text += f"Food expenditure: {food_total}\n"
            text += f"Non-food expenditure: {non_food_total}\n"

            # Top categories
            top_food = region_data.loc[food_items, "Value"].sort_values(ascending=False).head(5)
            top_non_food = region_data.loc[non_food_items, "Value"].sort_values(ascending=False).head(5)

            text += "\nTop food items:\n"
            for item, val in top_food.items():
                text += f" - {item}: {val}\n"

            text += "\nTop non-food items:\n"
            for item, val in top_non_food.items():
                text += f" - {item}: {val}\n"

            region_texts.append(text)

    return region_texts


# ----------------------------
# 2. Build Vector DB
# ----------------------------

def load_rag_index():
    texts = load_government_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text("\n\n".join(texts))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings)
    return vectordb


# ----------------------------
# 3. LLM + RAG Chain
# ----------------------------

def get_llm_chain(vectordb, extra_context=""):
    """
    Build final RAG + LLM chain with optional injected context
    (news + social media).
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
You are a retail strategy AI assistant.

Use ONLY the provided context from:
- Myanmar Government household spending tables
- Latest market news
- Social media consumption trends

CONTEXT:
{{context}}

LIVE MARKET SIGNALS:
{extra_context}

QUESTION:
{{question}}

Give a strategic recommendation in **4–6 bullet points**.
"""
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    class GroqLLMWrapper:
        """Small wrapper for Groq to behave like an LLM."""

        def __call__(self, prompt):
            resp = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message["content"]

    llm = GroqLLMWrapper()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )
