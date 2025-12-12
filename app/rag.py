from dotenv import load_dotenv
import os, json
from .db import get_db
import requests

load_dotenv()

OPENROUTER_KEY = os.getenv("API_KEY")
OPENROUTER_URL = os.getenv("BASE_URL")        

# -----------------------------
# 1. 获取阿里云 Embedding
# -----------------------------
def get_embedding(text: str):
    url = f"{OPENROUTER_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(data)
    # 返回向量
    return data["data"][0]["embedding"]




# -----------------------------
# 2. 插入文档
# -----------------------------
def add_document(text: str):
    db = get_db()
    embedding = get_embedding(text)

    doc_id = db.count_rows() + 1
    db.add([{
        "id": doc_id,
        "text": text,
        "embedding": embedding
    }])

    return {"status": "ok", "text": text}


# -----------------------------
# 3. 相似文档检索
# -----------------------------
def retrieve_similar_documents(query: str, top_k: int):
    db = get_db()
    query_embedding = get_embedding(query)

    results = db.search(query_embedding).limit(top_k).to_pandas()
    return results["text"].tolist()
