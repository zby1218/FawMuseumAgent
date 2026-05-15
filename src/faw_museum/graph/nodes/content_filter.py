from ...config.loader import load_blocked_keywords
from ..state import RobotState

_BLOCK_RESPONSES = {
    "political":     "这个话题我不太方便讨论，有什么其他可以帮您的吗？",
    "gambling_drug": "抱歉，我无法回答这类问题。有什么其他可以帮您的吗？",
    "competitor":    "我对其他品牌不太了解，不过我可以介绍一下我们一汽的产品，您感兴趣吗？",
}


def _check(text: str) -> str | None:
    """返回触发的拦截类型，放行返回 None。"""
    kw = load_blocked_keywords()

    for category in ("political", "gambling_drug"):
        if any(word in text for word in kw.get(category, [])):
            return category

    brands = kw.get("competitor_brands", [])
    comparisons = kw.get("competitor_comparison", [])
    if any(b in text for b in brands) and any(c in text for c in comparisons):
        return "competitor"

    return None


def content_filter_node(state: RobotState) -> dict:
    text = state["messages"][-1].content
    blocked = _check(text)
    if blocked:
        return {
            "intent_type": None,
            "response_text": _BLOCK_RESPONSES[blocked],
        }
    return {}