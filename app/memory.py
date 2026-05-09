"""
滚动分层压缩记忆系统
====================

三层结构：
  Layer 1 — Working Memory:  最近 N 轮完整保留
  Layer 2 — Compressed Memory: 滚动压缩摘要（每 N 轮一次）
  Layer 3 — Long-term Facts:   跨会话关键事实（从旧摘要提取）

持久化：SQLite，按 thread_id 分区，服务重启不丢失。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import config
from app.prompts import MEMORY_COMPRESS_PROMPT, MEMORY_EXTRACT_FACTS_PROMPT

memory_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)

# --------------------------------------------------
# Data models
# --------------------------------------------------

@dataclass
class ConversationTurn:
    role: str        # "user" | "ai"
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class CompressedBlock:
    summary: str
    key_topics: list[str]
    turn_range: tuple[int, int]  # (start_turn, end_turn)
    compressed_at: float = field(default_factory=time.time)

@dataclass
class LongTermFact:
    fact: str
    category: str       # "用户信息" | "系统配置" | "IT问题" | "偏好设置"
    source_turns: list[int]
    is_active: bool = True
    last_updated: float = field(default_factory=time.time)


# --------------------------------------------------
# SQLite helpers
# --------------------------------------------------

def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(config.MEMORY_DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    return db


def _init_db():
    db = _get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            PRIMARY KEY (thread_id, turn_index)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS compressed (
            thread_id TEXT NOT NULL,
            block_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            key_topics TEXT NOT NULL,
            turn_start INTEGER NOT NULL,
            turn_end INTEGER NOT NULL,
            compressed_at REAL NOT NULL,
            PRIMARY KEY (thread_id, block_index)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            thread_id TEXT NOT NULL,
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'IT问题',
            source_turns TEXT NOT NULL DEFAULT '[]',
            is_active INTEGER NOT NULL DEFAULT 1,
            last_updated REAL NOT NULL
        )
    """)
    db.commit()
    db.close()


_init_db()


# --------------------------------------------------
# LLM helpers
# --------------------------------------------------

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# --------------------------------------------------
# Memory Manager
# --------------------------------------------------

class MemoryManager:
    """管理单个 thread 的三层记忆。

    用法:
        mem = MemoryManager("thread-uuid")
        mem.add_turn("user", "VPN怎么连？")
        mem.add_turn("ai", "先装 GlobalProtect...")
        context = mem.get_context_for_llm()
        # context 可以直接注入 Planner/Executor Prompt
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id

    # ---- Layer 1: Working Memory ----

    def add_turn(self, role: str, content: str):
        """添加一轮对话。自动触发压缩检查。"""
        db = _get_db()
        turn_idx = self._count_turns(db)
        db.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
            (self.thread_id, turn_idx, role, content, time.time()),
        )
        db.commit()
        db.close()

        # 检查是否需要滚动压缩
        total = turn_idx + 1
        if total > 0 and total % config.MEMORY_COMPRESSION_INTERVAL == 0:
            self._compress_oldest_block()

    def _count_turns(self, db: sqlite3.Connection) -> int:
        row = db.execute(
            "SELECT COUNT(*) FROM conversations WHERE thread_id = ?",
            (self.thread_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_working_memory(self, limit: int | None = None) -> list[dict]:
        """获取最近 N 轮完整对话。"""
        limit = limit if limit is not None else config.MEMORY_WORKING_TURNS
        db = _get_db()
        rows = db.execute(
            "SELECT role, content FROM conversations "
            "WHERE thread_id = ? ORDER BY turn_index DESC LIMIT ?",
            (self.thread_id, limit),
        ).fetchall()
        db.close()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # ---- Layer 2: Compressed Memory ----

    def _compress_oldest_block(self):
        """将最早的 N 轮对话压缩为一条摘要，删除原文，存入压缩层。"""
        db = _get_db()
        total = self._count_turns(db)
        if total <= config.MEMORY_WORKING_TURNS:
            db.close()
            return

        # 超出 working memory 的最早 N 轮
        overflow = total - config.MEMORY_WORKING_TURNS
        block_size = min(overflow, config.MEMORY_COMPRESSION_INTERVAL)
        if block_size <= 0:
            db.close()
            return

        # 取最早 block_size 轮
        rows = db.execute(
            "SELECT turn_index, role, content FROM conversations "
            "WHERE thread_id = ? ORDER BY turn_index ASC LIMIT ?",
            (self.thread_id, block_size),
        ).fetchall()
        db.close()

        if not rows:
            return

        turn_min = rows[0][0]
        turn_max = rows[-1][0]
        conv_text = "\n".join(f"[{r}]: {c[:300]}" for _, r, c in rows)

        # 取已有事实
        facts = self.get_active_facts()
        facts_text = "\n".join(f"- {f.fact}" for f in facts) if facts else "（无）"

        # LLM 压缩
        prompt = MEMORY_COMPRESS_PROMPT.format(conversation=conv_text, existing_facts=facts_text)
        try:
            resp = memory_llm.invoke(prompt)
            data = _extract_json(str(resp.content)) or {}
        except Exception:
            data = {}

        summary = str(data.get("summary", conv_text[:200]))
        topics = [str(t)[:24] for t in (data.get("key_topics") or [])][:6]
        extracted = data.get("facts_extracted") or []

        # 存压缩块
        db = _get_db()
        block_idx = db.execute(
            "SELECT COALESCE(MAX(block_index), -1) + 1 FROM compressed WHERE thread_id = ?",
            (self.thread_id,),
        ).fetchone()[0]

        db.execute(
            "INSERT INTO compressed VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.thread_id, block_idx, summary, json.dumps(topics, ensure_ascii=False),
             turn_min, turn_max, time.time()),
        )

        # 删除已压缩的原文
        db.execute(
            "DELETE FROM conversations WHERE thread_id = ? AND turn_index BETWEEN ? AND ?",
            (self.thread_id, turn_min, turn_max),
        )

        # 限制压缩块数量
        db.execute(
            "DELETE FROM compressed WHERE thread_id = ? AND block_index NOT IN "
            "(SELECT block_index FROM compressed WHERE thread_id = ? "
            "ORDER BY block_index DESC LIMIT ?)",
            (self.thread_id, self.thread_id, config.MEMORY_MAX_COMPRESSED),
        )

        db.commit()
        db.close()

        # 提取长期事实
        if extracted:
            self._upsert_facts([LongTermFact(
                fact=e.get("fact", ""),
                category=e.get("category", "IT问题"),
                source_turns=list(range(turn_min, turn_max + 1)),
            ) for e in extracted if e.get("fact")])

    def get_compressed_blocks(self) -> list[CompressedBlock]:
        db = _get_db()
        rows = db.execute(
            "SELECT summary, key_topics, turn_start, turn_end, compressed_at "
            "FROM compressed WHERE thread_id = ? ORDER BY block_index ASC",
            (self.thread_id,),
        ).fetchall()
        db.close()
        return [CompressedBlock(
            summary=r[0],
            key_topics=json.loads(r[1]) if r[1] else [],
            turn_range=(r[2], r[3]),
            compressed_at=r[4],
        ) for r in rows]

    # ---- Layer 3: Long-term Facts ----

    def _upsert_facts(self, new_facts: list[LongTermFact]):
        db = _get_db()
        existing = self.get_active_facts()
        existing_texts = {f.fact for f in existing}

        for nf in new_facts:
            if not nf.fact or nf.fact in existing_texts:
                continue
            db.execute(
                "INSERT INTO facts (thread_id, fact, category, source_turns, is_active, last_updated) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (self.thread_id, nf.fact, nf.category,
                 json.dumps(nf.source_turns), time.time()),
            )

        # 控制数量上限
        db.execute(
            "DELETE FROM facts WHERE thread_id = ? AND fact_id NOT IN "
            "(SELECT fact_id FROM facts WHERE thread_id = ? "
            "ORDER BY last_updated DESC LIMIT ?)",
            (self.thread_id, self.thread_id, config.MEMORY_MAX_FACTS),
        )
        db.commit()
        db.close()

    def get_active_facts(self) -> list[LongTermFact]:
        db = _get_db()
        rows = db.execute(
            "SELECT fact, category, source_turns, is_active, last_updated "
            "FROM facts WHERE thread_id = ? AND is_active = 1 "
            "ORDER BY last_updated DESC",
            (self.thread_id,),
        ).fetchall()
        db.close()
        return [LongTermFact(
            fact=r[0], category=r[1],
            source_turns=json.loads(r[2]) if r[2] else [],
            is_active=bool(r[3]), last_updated=r[4],
        ) for r in rows]

    # ---- Assemble context for LLM ----

    def get_context_for_llm(self) -> str:
        """组装三层记忆为可注入 Prompt 的上下文字符串。"""
        parts = []

        # Layer 3: 长期事实
        facts = self.get_active_facts()
        if facts:
            parts.append("【用户历史画像】")
            for f in facts:
                parts.append(f"- [{f.category}] {f.fact}")

        # Layer 2: 压缩摘要
        blocks = self.get_compressed_blocks()
        if blocks:
            parts.append("\n【历史对话摘要】")
            for i, b in enumerate(blocks):
                topics = "、".join(b.key_topics) if b.key_topics else "通用"
                parts.append(f"阶段{i+1}[{topics}]: {b.summary}")

        # Layer 1: 最近完整对话
        recent = self.get_working_memory()
        if recent:
            parts.append("\n【近期对话】")
            for t in recent:
                role_label = "用户" if t["role"] == "user" else "AI"
                parts.append(f"{role_label}: {t['content'][:400]}")

        return "\n".join(parts) if parts else "（无历史对话）"

    def clear(self):
        """清除该 thread 的所有记忆。"""
        db = _get_db()
        db.execute("DELETE FROM conversations WHERE thread_id = ?", (self.thread_id,))
        db.execute("DELETE FROM compressed WHERE thread_id = ?", (self.thread_id,))
        db.execute("DELETE FROM facts WHERE thread_id = ?", (self.thread_id,))
        db.commit()
        db.close()
