from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_input: str
    intent: dict[str, Any] | None
    response_text: str