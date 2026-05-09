"""
GraphRAG 知识图谱模块
====================

用于解决跨文档和场景类问题的检索增强：
1. 离线：从所有文档中提取实体 + 关系，构建知识图谱
2. 在线：查询命中某文档时，沿图谱边找到相关文档，多跳检索
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import config
from app.prompts import ENTITY_EXTRACT_PROMPT, GRAPH_RETRIEVAL_PROMPT

graph_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)


# ==================================================
# Data models
# ==================================================

@dataclass
class Entity:
    name: str
    type: str         # "系统" | "软件" | "配置项" | "政策" | "操作"
    doc_id: str       # 所属文档

    def __hash__(self):
        return hash((self.name.lower(), self.doc_id))

@dataclass
class Relation:
    source: str       # entity name
    relation: str     # "需要" | "包含" | "关联" | "前置" | "替代"
    target: str       # entity name


@dataclass
class KnowledgeGraph:
    entities: dict[str, list[Entity]] = field(default_factory=lambda: defaultdict(list))
    relations: list[Relation] = field(default_factory=list)
    doc_index: dict[str, Entity] = field(default_factory=dict)  # entity_name → Entity

    def add_entity(self, e: Entity):
        self.entities[e.doc_id].append(e)
        key = e.name.lower()
        if key not in self.doc_index:
            self.doc_index[key] = e

    def add_relation(self, r: Relation):
        self.relations.append(r)

    def find_related_docs(self, entity_names: list[str]) -> set[str]:
        """跟给定的实体集合有关联的所有文档。"""
        related_docs = set()
        names_lower = {n.lower() for n in entity_names}

        for r in self.relations:
            if r.source.lower() in names_lower or r.target.lower() in names_lower:
                # 找到 source 和 target 各自在哪个文档里
                src_entity = self.doc_index.get(r.source.lower())
                tgt_entity = self.doc_index.get(r.target.lower())
                if src_entity:
                    related_docs.add(src_entity.doc_id)
                if tgt_entity:
                    related_docs.add(tgt_entity.doc_id)

        return related_docs

    def summary(self) -> str:
        """知识图谱摘要，供 LLM 推理用。"""
        lines = []
        for doc_id, entities in self.entities.items():
            doc_name = os.path.basename(doc_id)
            entity_str = ", ".join(f"{e.name}({e.type})" for e in entities[:8])
            lines.append(f"  {doc_name}: {entity_str}")
        for r in self.relations:
            lines.append(f"  {r.source} --[{r.relation}]--> {r.target}")
        return "\n".join(lines)


# Global graph instance
_knowledge_graph: KnowledgeGraph | None = None


# ==================================================
# Entity extraction (offline)
# ==================================================

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


def build_knowledge_graph(docs_dir: str | None = None) -> KnowledgeGraph:
    """离线构建：遍历所有文档，提取实体 + 关系。"""
    global _knowledge_graph

    docs_dir = docs_dir or config.KNOWLEDGE_BASE_DIR
    kg = KnowledgeGraph()

    import glob as _glob
    md_files = _glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        prompt = ENTITY_EXTRACT_PROMPT.format(content=content[:3000], doc_id=filepath)
        try:
            resp = graph_llm.invoke(prompt)
            data = _extract_json(str(resp.content)) or {}
        except Exception as e:
            print(f"⚠️ 实体提取失败 ({filepath}): {e}")
            continue

        # Parse entities
        for e in (data.get("entities") or []):
            kg.add_entity(Entity(
                name=str(e.get("name", "")).strip(),
                type=str(e.get("type", "未知")).strip(),
                doc_id=filepath,
            ))

        # Parse relations
        for r in (data.get("relations") or []):
            kg.add_relation(Relation(
                source=str(r.get("source", "")).strip(),
                relation=str(r.get("relation", "关联")).strip(),
                target=str(r.get("target", "")).strip(),
            ))

    _knowledge_graph = kg
    print(f"📊 知识图谱构建完成: {sum(len(v) for v in kg.entities.values())} 个实体, {len(kg.relations)} 条关系")
    return kg


def get_graph() -> KnowledgeGraph | None:
    global _knowledge_graph
    if _knowledge_graph is None:
        try:
            build_knowledge_graph()
        except Exception as e:
            print(f"⚠️ 知识图谱构建失败: {e}")
    return _knowledge_graph


# ==================================================
# Multi-hop retrieval (online)
# ==================================================

def get_related_docs(query: str, matched_entities: list[str]) -> list[str]:
    """给定查询和已匹配实体，沿图谱边找到相关文档列表。

    用于跨文档检索：当用户在问「远程办公需要准备什么」时，
    即使 MMR 只召回了 VPN 文档，图谱也能找到邮箱和 WiFi 相关文档。
    """
    kg = get_graph()
    if kg is None or not matched_entities:
        return []

    # 方法1: 基于图谱关系直接找
    graph_docs = kg.find_related_docs(matched_entities)

    # 方法2: LLM 推理补充（图谱可能不完整）
    if graph_docs:
        prompt = GRAPH_RETRIEVAL_PROMPT.format(
            query=query,
            graph_summary=kg.summary(),
            matched_entities=", ".join(matched_entities),
        )
        try:
            resp = graph_llm.invoke(prompt)
            data = _extract_json(str(resp.content)) or {}
            llm_docs = [d for d in (data.get("additional_docs") or []) if isinstance(d, str)]
        except Exception:
            llm_docs = []

        # 合并去重
        all_docs = set()
        for d in graph_docs:
            if d:
                all_docs.add(os.path.basename(d))
        for d in llm_docs:
            if d:
                all_docs.add(d)

        return sorted(all_docs)

    return []
