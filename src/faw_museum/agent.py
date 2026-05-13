from typing import Any
from langchain_core.messages import HumanMessage
from .graph.graph import build_graph
from .graph.state import create_initial_state


class MuseumAgent:
    def __init__(self) -> None:
        self.graph = build_graph()

    def process(self, robot_id: str, user_input: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": robot_id}}

        existing = self.graph.get_state(config)
        if existing.values:
            input_state = {"messages": [HumanMessage(content=user_input)]}
        else:
            input_state = create_initial_state(robot_id, user_input)

        result = self.graph.invoke(input_state, config=config)
        return {
            "intent_type": result.get("intent_type"),
            "response": result.get("response_text", ""),
        }