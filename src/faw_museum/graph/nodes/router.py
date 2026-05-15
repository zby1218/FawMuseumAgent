import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ...config.loader import load_llm_config, load_system_prompts
from ...llm import get_llm
from ..state import RobotState

logger = logging.getLogger(__name__)

_VALID_INTENTS = frozenset({"chitchat", "knowledge", "action", "mute"})

# 格式约束：开发维护，不进 system_prompts.yaml
_FORMAT_PROMPT = """\
请将用户输入分类为以下意图之一：
- chitchat：闲聊、问候、日常对话（示例："你好"、"你叫什么名字"、"讲个故事"）
- knowledge：询问展品、一汽历史、车型参数等知识性问题（示例："红旗H9续航多少"、"这辆车哪年生产的"）
- action：要求机器人移动或执行动作（示例："过来"、"带我去展区"、"点头"）
- mute：要求机器人安静或停止说话（示例："你安静一会儿"、"别说了"）

只输出 JSON，格式：{"intent": "chitchat"|"knowledge"|"action"|"mute", "reason": "简要说明"}
不要输出任何其他内容。"""


async def router_node(state: RobotState) -> dict:
    prompts = load_system_prompts()
    system = prompts["brand_identity"] + "\n\n" + _FORMAT_PROMPT
    user_text = state["messages"][-1].content

    infer = load_llm_config()["inference"]["classify"]
    response = await get_llm().ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user_text)],
        **infer,
    )
    return {"intent_type": _parse_intent(response.content)}


def _parse_intent(raw: str) -> str:
    try:
        intent = json.loads(raw.strip()).get("intent", "").lower()
        if intent in _VALID_INTENTS:
            return intent
    except (json.JSONDecodeError, AttributeError):
        pass
    logger.warning("router: 无法解析输出 %r，fallback 到 chitchat", raw)
    return "chitchat"