from fastapi import HTTPException
from .coref import need_coref, resolve_coreference
from app.chat_manager import append_massage,get_last_turns
from .rag import get_embedding
from .retriever import embedding_retriever, bm25_retriever, embedding_retriever_with_history
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MAX_TOKENS=800  

def count_tokens(text: str) -> int:
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoder.encode(text)
    return len(tokens)

#构建RAG context
def build_context(query: str, top_k: int = 3, conversation_id: str = None):
    # Step 1: 检索 top_k 文档
    docs = embedding_retriever_with_history(query, top_k, conversation_id)

    # Step 2: 提取文本，保证都是字符串
    doc_texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in docs]

    # Step 3: 添加历史轮次 context
    history_texts = []
    if conversation_id:
        from .chat_manager import get_last_turns
        last_turns = get_last_turns(conversation_id)
        for turn in last_turns:
            history_texts.append(f"{turn['role'].capitalize()}: {turn['content']}")

    # Step 4: 拼接 docs + 历史 context
    context_lines = doc_texts + history_texts
    context = "\n".join(context_lines)

    # Step 5: token 截断
    while count_tokens(context) > MAX_TOKENS:
        if len(doc_texts) > 0:
            # 先截掉最旧的 doc
            doc_texts.pop(0)
            context_lines = doc_texts + history_texts
        else:
            # docs 已经空了，只剩历史 context，不再截断
            break
        context = "\n".join(context_lines)

    return context

def sanitize_history(history):
    sanitized = []
    for turn in history:
        if turn["role"] == "user" and need_coref(turn["content"]):
            sanitized.append({
                "role": "user",
                "content": resolve_coreference(turn["content"], history)
            })
        else:
            sanitized.append(turn)
    return sanitized


def chat(query: str, conversation_id: str=None, top_k: int=1):
    
    history = []
    if conversation_id:
        history = sanitize_history(get_last_turns(conversation_id))

    resolved_query = query
    if conversation_id and need_coref(query):
        resolved_query = resolve_coreference(query, history)

    context = build_context(
        resolved_query,
        top_k,
        conversation_id=None  
    )

    history_block = ""
    if history:
        history_block = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in history
        )

    messages = [
    {
        "role": "system",
        "content": f"""
你是一个智能问答助手。

【对话历史（优先级高）】
{history_block}

【参考资料】
{context}

回答要求：
- 本轮问题中的代词必须指向历史中最近出现的明确技术实体
- 不得引入历史或问题中未出现的新框架
"""
    },
    {"role": "user", "content": resolved_query}
]


    try:
        resp = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # 注意模型名
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )

        answer = resp.choices[0].message.content
        if conversation_id:
            from .chat_manager import append_massage
            append_massage(conversation_id, "user", query)
            append_massage(conversation_id, "assistant", answer)
        return {"answer": answer, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))