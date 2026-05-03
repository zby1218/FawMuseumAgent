from typing import Any
from .graph.builder import build_graph
from .graph.state import AgentState


class MuseumAgent:
    def __init__(self) -> None:
        self.graph = build_graph()

    def process(self, user_input: str) -> dict[str, Any]:
        result = self.graph.invoke(
            AgentState(user_input=user_input, intent=None, response_text="")
        )
        return {
            "intent": result.get("intent"),
            "response": result.get("response_text", ""),
        }