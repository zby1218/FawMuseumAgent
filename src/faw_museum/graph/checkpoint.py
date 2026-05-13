from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


def get_checkpointer() -> BaseCheckpointSaver:
    # 迁移 PostgreSQL 时只改这一处：
    # import os
    # from langgraph.checkpoint.postgres import PostgresSaver
    # return PostgresSaver.from_conn_string(os.environ["POSTGRES_URI"])
    return MemorySaver()