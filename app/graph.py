"""
LangGraph 图定义 — 通用型 Agent
图结构: analyze_question → generate_answer → END
"""

import os
import sqlite3
import warnings

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import AgentState
from app.config import config
from app.nodes import analyze_question, retrieve_kb, generate_answer

# SQLite 连接复用
_sqlite_conn: sqlite3.Connection | None = None
_checkpointer_cache: tuple[str, object] | None = None


def _get_checkpointer():
    global _sqlite_conn, _checkpointer_cache
    path = config.LANGGRAPH_CHECKPOINT_SQLITE
    if _checkpointer_cache and _checkpointer_cache[0] == path:
        return _checkpointer_cache[1]

    if path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            warnings.warn(
                "已设置 LANGGRAPH_CHECKPOINT_SQLITE 但未安装 langgraph-checkpoint-sqlite，"
                "回退到 MemorySaver。请执行: pip install langgraph-checkpoint-sqlite",
                stacklevel=2,
            )
            saver = MemorySaver()
        else:
            abs_path = os.path.abspath(path)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _sqlite_conn = sqlite3.connect(abs_path, check_same_thread=False)
            saver = SqliteSaver(_sqlite_conn)
    else:
        saver = MemorySaver()

    _checkpointer_cache = (path, saver)
    return saver


def build_graph(with_memory: bool = True):
    """
    构建通用型 Agent 图

    图结构:
        ┌──────────────────┐
        │ analyze_question │  (分析问题领域/类型)
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ generate_answer  │  (RAG 检索 + LLM 生成回答)
        └────────┬─────────┘
                 │
                 ▼
                END
    """

    graph = StateGraph(AgentState)

    graph.add_node("analyze_question", analyze_question)
    graph.add_node("retrieve_kb", retrieve_kb)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("analyze_question")
    graph.add_edge("analyze_question", "retrieve_kb")
    graph.add_edge("retrieve_kb", "generate_answer")
    graph.add_edge("generate_answer", END)

    if with_memory:
        return graph.compile(checkpointer=_get_checkpointer())
    else:
        return graph.compile()


agent = build_graph(with_memory=True)
