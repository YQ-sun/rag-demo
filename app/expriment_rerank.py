from .retriever import embedding_retriever
from .reranker import rerank_responses
import difflib
from sentence_transformers import SentenceTransformer, util
from nltk import sent_tokenize
import re
from typing import List

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ---------- 简单中英文分句（无 NLTK） ----------
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'[。！？.!?]\s*', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ---------- gold sentence 匹配 ----------
def match_gold_sentence(
    gold_sentences: List[str],
    ranked_sentences: List[str],
    threshold: float = 0.4,
    top_k: int = 5
) -> int:
    """
    返回 gold 命中的最优排名（1-based），未命中返回 -1
    """
    for i, cand in enumerate(ranked_sentences[:top_k]):
        for gold in gold_sentences:
            ratio = difflib.SequenceMatcher(None, gold, cand).ratio()
            if ratio >= threshold:
                return i + 1
    return -1


# ---------- 实验主逻辑 ----------
def run_experiment(
    dataset: List[dict],
    retrieve_k: int = 20,
    rerank_k: int = 5,
    threshold: float = 0.4
):
    hit1 = hit3 = hit5 = 0

    for idx, item in enumerate(dataset):
        query = item["query"]
        gold_docs = item["gold_docs"]

        # ====== Retriever ======
        retrieved_docs = embedding_retriever(query, top_k=retrieve_k)

        # 文档 → 句子
        candidate_sentences = []
        for doc in retrieved_docs:
            candidate_sentences.extend(split_sentences(doc))

        candidate_sentences = list(set(candidate_sentences))

        print(f"\n[Query {idx+1}] {query}")
        print(f"Candidate sentences: {len(candidate_sentences)}")

        if not candidate_sentences:
            continue

        # ====== Reranker ======
        ranked = rerank_responses(query, candidate_sentences)
        ranked_sentences = [text for text, _ in ranked]

        print("\n[Reranked top 5]")
        for i, (text, score) in enumerate(ranked[:5], 1):
            print(f"{i}: score={score:.4f} | {text[:80]}")

        # ====== Gold matching ======
        gold_sentences = []
        for g in gold_docs:
            gold_sentences.extend(split_sentences(g))

        rank = match_gold_sentence(
            gold_sentences,
            ranked_sentences,
            threshold=threshold,
            top_k=rerank_k
        )

        if rank == 1:
            hit1 += 1
            hit3 += 1
            hit5 += 1
        elif 1 < rank <= 3:
            hit3 += 1
            hit5 += 1
        elif 3 < rank <= 5:
            hit5 += 1
        else:
            print("Gold doc NOT found in top 5")
            for g in gold_sentences[:2]:
                print("Gold:", g[:80])

    total = len(dataset)
    print("\n========== FINAL RESULT ==========")
    print(f"Top-1 Accuracy: {hit1 / total:.2%}")
    print(f"Top-3 Accuracy: {hit3 / total:.2%}")
    print(f"Top-5 Accuracy: {hit5 / total:.2%}")


# ---------- 示例入口 ----------
if __name__ == "__main__":
    # 示例 dataset，你替换成真实数据即可
    dataset = [
    {"query":"FastAPI 是什么？","gold_docs":["FastAPI 是一个用于构建高性能 Python Web API 的框架，基于 Starlette 和 Pydantic。"]},
    {"query": "FastAPI 和 Flask 有什么区别？", "gold_docs": ["FastAPI 原生支持异步、高性能和自动生成 OpenAPI 文档，Flask 更轻量但需手动扩展。"]},
    {"query": "如何在 FastAPI 中处理请求？", "gold_docs": ["在 FastAPI 中，可以通过定义路径操作函数来处理 HTTP 请求，这些函数使用装饰器如 @app.get() 或 @app.post() 来指定路由。"]},
    {"query": "FastAPI 有哪些主要特性？", "gold_docs": ["FastAPI 的主要特性包括：高性能、自动生成 API 文档、类型提示支持、异步支持等。"]},
    {"query": "如何在 FastAPI 中进行数据验证？", "gold_docs": ["FastAPI 使用 Pydantic 模型进行数据验证，定义请求体时可以指定 Pydantic 模型，FastAPI 会自动验证传入的数据。"]},
    {"query": "FastAPI 支持哪些数据库？", "gold_docs": ["FastAPI 可以与多种数据库一起使用，包括 SQL（如 SQLite、PostgreSQL、MySQL）和 NoSQL（如 MongoDB）数据库，通常通过 ORM（如 SQLAlchemy、Tortoise-ORM）进行集成。"]},
    {"query": "如何在 FastAPI 中实现身份验证？", "gold_docs": ["FastAPI 支持多种身份验证方法，包括 OAuth2、JWT、HTTP Basic Auth 等，可以使用 FastAPI 提供的安全工具或第三方库来实现。"]},
    {"query": "FastAPI 的性能如何？", "gold_docs": ["FastAPI 以其高性能著称，得益于其基于 Starlette 和 Uvicorn 的异步架构，能够处理大量并发请求，性能接近 Node.js 和 Go。"]},
    {"query": "如何在 FastAPI 中处理文件上传？", "gold_docs": ["在 FastAPI 中，可以使用 File 和 UploadFile 类型来处理文件上传，定义路径操作函数时将这些类型作为参数即可接收上传的文件。"]},
    {"query": "FastAPI 如何进行测试？", "gold_docs": ["FastAPI 提供了内置的测试客户端，可以使用 pytest 等测试框架编写测试用例，通过模拟请求来测试路径操作函数的行为。"]},
    {"query": "FastAPI 支持 WebSocket 吗？", "gold_docs": ["是的，FastAPI 原生支持 WebSocket，可以通过定义 WebSocket 路径操作函数来处理 WebSocket 连接和消息传递。"]},
    {"query": "如何在 FastAPI 中实现中间件？", "gold_docs": ["在 FastAPI 中，可以通过定义中间件函数并使用 @app.middleware 装饰器来实现中间件，用于处理请求和响应的预处理和后处理。"]},
    {"query": "FastAPI 支持异步编程吗？", "gold_docs": ["是的，FastAPI 完全支持异步编程，可以使用 async 和 await 关键字定义异步路径操作函数，以提高应用的并发性能。"]},
    {"query": "FastAPI 如何进行跨域资源共享（CORS）？", "gold_docs": ["FastAPI 提供了内置的 CORS 中间件，可以通过安装和配置 fastapi.middleware.cors.CORSMiddleware 来实现跨域资源共享。"]},
    {"query": "FastAPI 有哪些常用的扩展库？", "gold_docs": ["FastAPI 有许多常用的扩展库，如 FastAPI Users（用户管理）、FastAPI JWT Auth（JWT 认证）、FastAPI Mail（邮件发送）等，帮助开发者快速构建功能丰富的应用。"]},
    {"query": "如何在 FastAPI 中处理表单数据？", "gold_docs": ["在 FastAPI 中，可以使用 Form 类型来处理表单数据，定义路径操作函数时将 Form 类型作为参数即可接收表单提交的数据。"]},
    {"query": "FastAPI 支持模板引擎吗？", "gold_docs": ["是的，FastAPI 支持多种模板引擎，如 Jinja2，可以通过集成模板引擎来渲染 HTML 页面。"]},
    {"query": "FastAPI 如何进行日志记录？", "gold_docs": ["FastAPI 使用标准的 Python 日志库 logging 进行日志记录，可以通过配置日志记录器来控制日志的格式和输出位置。"]},
    {"query": "FastAPI 支持后台任务吗？", "gold_docs": ["是的，FastAPI 支持后台任务，可以使用 BackgroundTasks 类来定义和执行后台任务，例如发送电子邮件或处理长时间运行的操作。"]},
    {"query": "FastAPI 如何进行依赖注入？", "gold_docs": ["FastAPI 提供了强大的依赖注入系统，可以通过在路径操作函数中使用 Depends 来声明依赖项，实现代码的模块化和可重用性。"]}
]

    run_experiment(dataset)