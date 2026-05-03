from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes.intent_router import intent_router_node
from .nodes.response import response_node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent_router", intent_router_node)
    g.add_node("response", response_node)
    g.add_edge(START, "intent_router")
    g.add_edge("intent_router", "response")
    g.add_edge("response", END)
    return g.compile()