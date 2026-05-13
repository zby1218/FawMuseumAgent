from typing import Any
from ..state import RobotState


def dispatch_node(state: RobotState) -> dict[str, Any]:
    return {}


def monitor_node(state: RobotState) -> dict[str, Any]:
    return {}
