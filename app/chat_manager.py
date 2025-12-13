from typing import List, Dict
from collections import defaultdict

conversation_store: Dict[str, List[Dict]] = defaultdict(list)
MAX_HISTORY_TURNS=2

def append_massage(conversation_id: str, role: str, content: str):
    history = conversation_store[conversation_id]
    history.append({"role": role, "content": content})

def get_last_turns(conversation_id: str, n: int = MAX_HISTORY_TURNS):
    history = conversation_store.get(conversation_id, [])
    return history[-(n * 2):]  # 每个回合包含用户和助手的消息