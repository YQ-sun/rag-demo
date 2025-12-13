from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

PRONOUNS = ["它", "他", "她", "这个", "那个", "该", "上述", "前面", "这种"]

def need_coref(query: str) -> bool:
    """简单规则判断是否需要指代消解"""
    return any(p in query for p in PRONOUNS)


def resolve_coreference(
    query: str,
    conversation_history: list[dict]
) -> str:
    """
    使用 LLM 将含代词的问题改写为自洽问题
    """

    # 只取最近 2 轮（避免 prompt 过长）
    history_text = ""
    for h in conversation_history[-4:]:
        history_text += f"{h['role']}: {h['content']}\n"

    prompt = f"""
你是一个对话理解模块，负责“指代消解（coreference resolution）”。

任务：
- 用户的问题中可能包含“它 / 这个 / 该框架 / 上述方法”等代词
- 请根据【对话历史】，将问题改写为一个“指代明确、自洽的问题”
- 不要回答问题，只返回改写后的问题
- 如果不需要改写，原样返回
如果问题是【定义类问题】，可以基于通用知识回答
如果问题是【对比类问题】（如 A 和 B 的区别）：
- 允许引入通用背景知识进行对比
- 但必须明确指出哪些信息来自通用知识
不得引入与问题无关的第三实体
若确实无法判断，再说明不确定


【对话历史】
{history_text}

【当前问题】
{query}

【改写后的问题】
"""

    resp = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    return resp.choices[0].message.content.strip()