from typing import Any
from ..state import RobotState


def mute_node(state: RobotState) -> dict[str, Any]:
    return {}


def wake_up_node(state: RobotState) -> dict[str, Any]:
    return {}


def discard_node(state: RobotState) -> dict[str, Any]:
    return {}
