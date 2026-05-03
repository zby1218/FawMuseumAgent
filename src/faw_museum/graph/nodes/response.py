from typing import Any
from ..state import AgentState


def response_node(state: AgentState) -> dict[str, Any]:
    intent = state.get("intent")

    if not intent or intent["type"] == "unknown":
        return {"response_text": "抱歉，我没有理解您的意思。"}

    return {
        "response_text": (
            f"[{intent['type']}] 已识别，"
            f"来源: {intent['source']}，"
            f"置信度: {intent['confidence']:.2f}"
        )
    }