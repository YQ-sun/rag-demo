from typing import List
from .rag import get_embedding
from .db import get_db
import os
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import QueryParser
import pandas as pd
from .reranker import rerank_responses

BM25_INDEX_DIR = "data/bm25_index"

def embedding_retriever_initial(query: str, top_k: int = 5):
    db = get_db()
    query_embedding = get_embedding(query)

    results = db.search(query_embedding).limit(top_k).to_pandas()
    # 只返回 text + score，不返回 embedding
    return [
        {"text": row["text"], "score": float(row["_distance"])}
        for _, row in results.iterrows()
    ]

def embedding_retriever(query: str, top_k: int = 5,reranker_method:str |None=None) -> List[str]:
    candidate_docs = embedding_retriever_initial(query, top_k=top_k*3)
    if reranker_method:
        ranked = rerank_responses(
            user_query=query,
            candidate_responses=candidate_docs,
            method=reranker_method
        )
        return [doc for doc, score in ranked[:top_k]]
    else:
        return [doc["text"] for doc in candidate_docs[:top_k]]

def embedding_retriever_with_history(query:str,top_k:int=3, conversation_id:str=None):
    query_embedding= get_embedding(query)
    db= get_db()
    docs= db.search(query_embedding).limit(top_k).to_pandas()

    if conversation_id:
        from .chat_manager import get_last_turns
        last_turns= get_last_turns(conversation_id)
        for turn in last_turns:
            if turn['role'] == 'assistant':
                # 把每条历史回答当作一条额外文档
                docs = pd.concat([docs, pd.DataFrame([{"text": turn['content']}])], ignore_index=True)

    return docs.head(top_k).to_dict(orient="records")


def build_bm25_index():
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)
    schema= Schema(id=ID(stored=True, unique=True), text=TEXT(stored=True))
    if not exists_in(BM25_INDEX_DIR):
        ix = create_in(BM25_INDEX_DIR, schema)
    else:
        ix = open_dir(BM25_INDEX_DIR)
    writer = ix.writer()
    table = get_db().to_pandas()
    for idx, row in table.iterrows():
        writer.add_document(id=str(row["id"]), text=row["text"])
    writer.commit()
    print("BM25 index built successfully.")

def bm25_retriever(query: str, top_k: int = 5):
    if not exists_in(BM25_INDEX_DIR):
        raise []
    ix = open_dir(BM25_INDEX_DIR)
    parser= QueryParser("text", schema=ix.schema)
    q = parser.parse(query)
    results_list = []
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=top_k)
        for hit in results:
            results_list.append(hit["text"])
    
    return results_list