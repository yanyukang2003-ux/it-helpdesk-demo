"""Checkpoint 列表（无需调用 LLM）。"""

from app.graph import build_graph
from app.checkpoints import list_checkpoints


def test_list_checkpoints_empty_thread():
    g = build_graph(with_memory=True)
    assert list_checkpoints(g, "thread-that-never-ran-xyz", limit=5) == []
