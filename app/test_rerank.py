from .retriever import embedding_retriever

query = "FastAPI 是什么"
results = embedding_retriever(query, top_k=20)  # 增大初步 top-k
print("Retrieved documents:")
for i, doc in enumerate(results):
    print(f"{i+1}: {doc[:100]}")  # 打印前100字符
