"""
Checkpoint 列表与时间旅行分支（fork）

LangGraph 在带 checkpointer 时会在每个 super-step 自动落库；
本模块封装「列出历史」「从某 checkpoint 改写用户问题并继续执行」。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _short_content(content: Any, max_len: int = 400) -> Any:
    if isinstance(content, str) and len(content) > max_len:
        return content[:max_len] + "…"
    return content


def message_preview(m: BaseMessage) -> dict[str, Any]:
    return {
        "type": m.__class__.__name__,
        "id": getattr(m, "id", None),
        "content": _short_content(getattr(m, "content", None)),
    }


def _last_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "content", None):
            return str(msg.content)
    return ""


def _find_human_message(
    messages: list,
    *,
    human_message_id: str | None,
) -> HumanMessage:
    if human_message_id:
        for m in messages:
            if isinstance(m, HumanMessage) and getattr(m, "id", None) == human_message_id:
                return m
        raise ValueError(f"未找到 id={human_message_id} 的 HumanMessage")

    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m

    raise ValueError("当前 checkpoint 的消息列表中没有 HumanMessage")


def list_checkpoints(
    compiled: Any,
    thread_id: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """返回某 thread 的 checkpoint 历史（新 → 旧），供前端展示与选择。"""
    cfg = {"configurable": {"thread_id": thread_id}}
    history = list(compiled.get_state_history(cfg, limit=limit))
    out: list[dict[str, Any]] = []
    for snap in history:
        writes = snap.metadata.get("writes") if snap.metadata else None
        writes_keys = list(writes.keys()) if isinstance(writes, dict) else None
        conf = snap.config.get("configurable", {})
        parent = snap.parent_config.get("configurable", {}) if snap.parent_config else {}
        out.append(
            {
                "checkpoint_id": conf.get("checkpoint_id"),
                "thread_id": conf.get("thread_id"),
                "parent_checkpoint_id": parent.get("checkpoint_id"),
                "next": list(snap.next),
                "step": (snap.metadata or {}).get("step"),
                "source": (snap.metadata or {}).get("source"),
                "writes_nodes": writes_keys,
                "created_at": snap.created_at,
                "messages": [message_preview(m) for m in snap.values.get("messages", [])],
            }
        )
    return out


def branch_from_checkpoint(
    compiled: Any,
    *,
    source_thread_id: str,
    from_checkpoint_id: str,
    new_human_content: str | None = None,
    human_message_id: str | None = None,
    override_state: dict[str, Any] | None = None,
    as_new_thread: bool = False,
    new_thread_id: str | None = None,
) -> dict[str, Any]:
    """
    在指定 checkpoint 上修改状态，然后继续执行图（会重跑该点之后的节点）。

    Parameters
    ----------
    new_human_content : 替换用户问题文本（None = 不替换）
    override_state    : 直接覆盖任意 AgentState 字段，例如 {"intent": "ticket_create"}
    as_new_thread     : 复制到新 thread_id 再执行（不污染原会话）

    建议在列表里选择 ``next`` 包含 ``\"classify\"`` 的检查点（用户输入已合并、尚未路由时）。
    """
    base_cfg: dict = {
        "configurable": {
            "thread_id": source_thread_id,
            "checkpoint_id": from_checkpoint_id,
            "checkpoint_ns": "",
        }
    }
    snap = compiled.get_state(base_cfg)
    if not snap or not snap.values:
        raise ValueError("无效的 thread_id 或 checkpoint_id")

    # ``get_state(checkpoint_id=...)`` returns values but drops next/metadata.
    # Walk history to recover them (small thread, cost is negligible).
    snap_meta = snap.metadata
    snap_next = snap.next
    snap_parent_cfg = snap.parent_config
    if snap_meta is None or not snap_next:
        thread_cfg = {"configurable": {"thread_id": source_thread_id}}
        for h in compiled.get_state_history(thread_cfg, limit=200):
            if h.config.get("configurable", {}).get("checkpoint_id") == from_checkpoint_id:
                snap_meta = h.metadata
                snap_next = h.next
                snap_parent_cfg = h.parent_config
                break

    warns: list[str] = []
    if snap_next and "classify" not in snap_next and not override_state:
        warns.append(
            "该 checkpoint 的 next 不含 classify，改写后重跑的行为可能与预期不符；"
            "建议选择 next 包含 classify 的检查点。"
        )

    # ── Build the state patch ──────────────────────────────────
    patch: dict[str, Any] = {}

    if new_human_content is not None:
        messages: list = list(snap.values.get("messages") or [])
        target = _find_human_message(messages, human_message_id=human_message_id)
        mid = target.id or str(uuid.uuid4())
        delta = HumanMessage(content=new_human_content, id=mid)
        patched_msgs = []
        for m in messages:
            if m is target:
                patched_msgs.append(delta)
            else:
                patched_msgs.append(m)
        patch["messages"] = patched_msgs

    if override_state:
        patch.update(override_state)

    # ── as_new_thread: seed a fresh thread with the patched state ──
    if as_new_thread:
        new_tid = (new_thread_id or str(uuid.uuid4())).strip()
        full_state = {k: v for k, v in snap.values.items()}
        full_state.update(patch)
        full_state.setdefault("intent", "")
        full_state.setdefault("retrieved_context", "")
        full_state.setdefault("ticket_info", {})
        full_state.setdefault("needs_human", False)
        full_state.setdefault("confidence", 1.0)
        new_cfg = {"configurable": {"thread_id": new_tid}}
        result = compiled.invoke(full_state, new_cfg)
        head = compiled.get_state(new_cfg)
        head_cid = head.config["configurable"].get("checkpoint_id") if head else None
        return {
            "thread_id": new_tid,
            "head_checkpoint_id": head_cid,
            "intent": result.get("intent", ""),
            "needs_human": result.get("needs_human", False),
            "reply": _last_ai_text(result.get("messages", [])),
            "analysis": result.get("analysis", ""),
            "needs_kb": bool(result.get("needs_kb", True)),
            "kb_sources": list(result.get("kb_sources") or []),
            "kb_preview": (result.get("retrieved_context") or "")[:140],
            "warnings": warns,
        }

    # ── Same-thread fork via update_state ─────────────────────
    # as_node must be the node whose write produced this checkpoint, so
    # LangGraph re-evaluates outgoing edges with the patched state.
    #
    # Derivation order:
    #   1. metadata.writes (most reliable when present)
    #   2. parent checkpoint's `next` (the node that was about to run = the node
    #      that just ran by the time we got here)
    #   3. fallback to the only known terminal sentinel
    writes = (snap_meta or {}).get("writes") or {}
    as_node: str | None = next(iter(writes.keys()), None)

    if not as_node:
        parent_cid = (snap_parent_cfg or {}).get("configurable", {}).get("checkpoint_id")
        if parent_cid:
            thread_cfg = {"configurable": {"thread_id": source_thread_id}}
            for h in compiled.get_state_history(thread_cfg, limit=200):
                if h.config.get("configurable", {}).get("checkpoint_id") == parent_cid:
                    if h.next:
                        as_node = list(h.next)[0]
                    break

    if not as_node:
        as_node = "__start__"

    if patch:
        fork_cfg = compiled.update_state(base_cfg, patch, as_node=as_node)
    else:
        # Nothing to patch — just re-run from the existing checkpoint as-is.
        fork_cfg = base_cfg

    result = compiled.invoke(None, fork_cfg)
    head = compiled.get_state({"configurable": {"thread_id": source_thread_id}})
    head_cid = head.config["configurable"].get("checkpoint_id") if head else None
    return {
        "thread_id": source_thread_id,
        "head_checkpoint_id": head_cid,
        "intent": result.get("intent", ""),
        "needs_human": result.get("needs_human", False),
        "reply": _last_ai_text(result.get("messages", [])),
        "analysis": result.get("analysis", ""),
        "needs_kb": bool(result.get("needs_kb", True)),
        "kb_sources": list(result.get("kb_sources") or []),
        "kb_preview": (result.get("retrieved_context") or "")[:140],
        "warnings": warns,
    }
