from typing import Optional, Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages


class SkillStep(TypedDict):
    skill: str          # 合法值由 dispatch 节点在 SUPPORTED_SKILLS 里校验
    params: dict
    description: str


class RobotState(TypedDict):
    # ── 会话标识 ──────────────────────────────────────────
    robot_id: str

    # ── 消息历史 ──────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── 系统状态（State Guard 读，各执行节点写）────────────
    silent_mode: bool
    execution_status: Literal["idle", "executing", "waiting_input", "waiting_confirm"]
    # idle            — 空闲，正常接受输入
    # executing       — 正在执行 skill，新输入全部 discard
    # waiting_input   — 槽位不全，等待用户补充信息
    # waiting_confirm — 高风险操作，等待用户二次确认
    current_task_type: Optional[str]  # 合法值与 SkillStep.skill 一致，由 dispatch 维护

    # ── 意图分类（Router 写，Content Filter 条件边读）────
    intent_type: Optional[Literal["chitchat", "knowledge", "action", "mute"]]

    # ── 任务规划（intent_parse 写，dispatch/validate 读）───
    skill_sequence: list[SkillStep]
    current_step: int

    # ── 执行结果（ROS 回调写，monitor 读）─────────────────
    last_ros_result: Optional[dict]

    # ── 输出 ──────────────────────────────────────────────
    response_text: str


def create_initial_state(robot_id: str, user_input: str) -> RobotState:
    """仅在新会话第一次 invoke 时调用。
    后续多轮对话只传 {"messages": [HumanMessage(content=user_input)]}，
    state 从 checkpoint 恢复，不重新初始化。
    """
    return RobotState(
        robot_id=robot_id,
        messages=[HumanMessage(content=user_input)],
        silent_mode=False,
        execution_status="idle",
        current_task_type=None,
        intent_type=None,
        skill_sequence=[],
        current_step=0,
        last_ros_result=None,
        response_text="",
    )