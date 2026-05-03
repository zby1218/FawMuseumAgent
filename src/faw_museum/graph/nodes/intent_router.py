from typing import Any
from ..state import AgentState
from ...config.loader import load_intents


def intent_router_node(state: AgentState) -> dict[str, Any]:
    """
    意图路由节点（当前阶段：关键词匹配占位）
    后续替换为：关键词拦截 → LLM 分类器
    """
    user_input = state["user_input"]
    config = load_intents()
    intents_cfg = config.get("intents", {})

    text = user_input.lower()

    for intent_type, cfg in intents_cfg.items():
        keywords = cfg.get("keywords", [])
        if any(kw in text for kw in keywords):
            return {
                "intent": {
                    "type": intent_type,
                    "confidence": 0.6,
                    "slots": {},
                    "source": "keyword",
                }
            }

    return {
        "intent": {
            "type": "unknown",
            "confidence": 0.0,
            "slots": {},
            "source": "keyword",
        }
    }