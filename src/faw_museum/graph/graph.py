from langgraph.graph import StateGraph, START, END
from .state import RobotState
from .checkpoint import get_checkpointer
from .nodes.guard import state_guard_node
from .nodes.router import router_node
from .nodes.content_filter import content_filter_node
from .nodes.chitchat import chitchat_node
from .nodes.knowledge import knowledge_node
from .nodes.intent import intent_parse_node
from .nodes.confirm import confirm_node, handle_confirmation_node
from .nodes.dispatch import dispatch_node, monitor_node
from .nodes.mute import mute_node, wake_up_node, discard_node

# 唤醒词列表（后续迁移到 config/wake_words.yaml）
_WAKE_WORDS = ["再聊", "醒醒", "醒来", "你好", "喂"]


def _guard_router(state: RobotState) -> str:
    """
    第一个节点的条件边，决定整个 invoke 的走向。
    优先级：静默模式 > 执行中 > 等待确认 > 正常流程
    """
    if state["silent_mode"]:
        last_msg = state["messages"][-1].content
        if any(w in last_msg for w in _WAKE_WORDS):
            return "wake_up"
        return "discard"

    if state["execution_status"] == "executing":
        return "discard"

    if state["execution_status"] == "waiting_confirm":
        return "handle_confirmation"

    return "router"


def _content_filter_router(state: RobotState) -> str:
    """
    内容过滤后的路由。
    content_filter_node 拦截时将 intent_type 置为 None 并写入 response_text。
    放行时 intent_type 保持 router_node 写入的值。
    """
    intent = state.get("intent_type")
    if intent is None:
        return "end"
    return intent  # "chitchat" / "knowledge" / "action" / "mute"


def _confirm_router(state: RobotState) -> str:
    """
    confirm_node 触发 interrupt() 时将 execution_status 置为 waiting_confirm。
    未触发则直接进 dispatch。
    """
    if state["execution_status"] == "waiting_confirm":
        return "end"
    return "dispatch"


def _monitor_router(state: RobotState) -> str:
    """
    monitor_node 处理 ROS 回调结果后：
    - 还有下一步 → dispatch
    - 全部完成或失败（execution_status 回 idle）→ END
    """
    if state["current_step"] < len(state["skill_sequence"]):
        return "dispatch"
    return "end"


def build_graph():
    g = StateGraph(RobotState)

    # ── 节点注册 ──────────────────────────────────────────
    g.add_node("state_guard", state_guard_node)
    g.add_node("router", router_node)
    g.add_node("content_filter", content_filter_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("intent_parse", intent_parse_node)
    g.add_node("confirm", confirm_node)
    g.add_node("handle_confirmation", handle_confirmation_node)
    g.add_node("dispatch", dispatch_node)
    g.add_node("monitor", monitor_node)
    g.add_node("mute", mute_node)
    g.add_node("wake_up", wake_up_node)
    g.add_node("discard", discard_node)

    # ── 入口 ──────────────────────────────────────────────
    g.add_edge(START, "state_guard")

    # ── 模块1：状态守卫 ───────────────────────────────────
    g.add_conditional_edges(
        "state_guard",
        _guard_router,
        {
            "router": "router",
            "discard": "discard",
            "wake_up": "wake_up",
            "handle_confirmation": "handle_confirmation",
        },
    )

    # ── 模块2：意图路由 → 模块3：内容过滤 ────────────────
    g.add_edge("router", "content_filter")

    g.add_conditional_edges(
        "content_filter",
        _content_filter_router,
        {
            "chitchat": "chitchat",
            "knowledge": "knowledge",
            "action": "intent_parse",
            "mute": "mute",
            "end": END,
        },
    )

    # ── 模块4/5：闲聊 & 知识问答 ─────────────────────────
    g.add_edge("chitchat", END)
    g.add_edge("knowledge", END)

    # ── 模块9：静默控制 ───────────────────────────────────
    g.add_edge("mute", END)
    g.add_edge("wake_up", END)
    g.add_edge("discard", END)

    # ── handle_confirmation（等待确认期间新输入的处理）─────
    # 骨架阶段直接结束；真实实现需 Command(resume=...) 恢复 invoke1
    g.add_edge("handle_confirmation", END)

    # ── 模块6→7→8：意图解析 → 确认 → 分发 → 监控 ─────────
    g.add_edge("intent_parse", "confirm")

    g.add_conditional_edges(
        "confirm",
        _confirm_router,
        {
            "dispatch": "dispatch",
            "end": END,
        },
    )

    g.add_edge("dispatch", "monitor")

    g.add_conditional_edges(
        "monitor",
        _monitor_router,
        {
            "dispatch": "dispatch",
            "end": END,
        },
    )

    return g.compile(checkpointer=get_checkpointer())
