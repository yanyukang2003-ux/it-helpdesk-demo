"""
节点定义 — 通用型 Agent
图结构: analyze_question → retrieve_kb → generate_answer → END

每个节点除了产出主结果，还会暴露"备选路径"给前端。
retrieve_kb 节点是否真正检索由 state.needs_kb 控制 —— 这让"是否查询知识库"
变成了思考流程里一个可见、可被用户覆盖的决策。
"""

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.state import AgentState
from app.config import config
from app.rag import retrieve_with_sources
from app.prompts import (
    ANALYZE_QUESTION_PROMPT,
    GENERAL_ANSWER_PROMPT,
    STYLE_HINTS,
)


# --------------------------------------------------
# LLM 实例
# --------------------------------------------------
router_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.4)
answer_llm = ChatOpenAI(model=config.ANSWER_MODEL, temperature=0.3)


# --------------------------------------------------
# 辅助函数
# --------------------------------------------------

def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            content = msg.content
            if isinstance(content, list):
                return " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            return str(content)
    return ""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


# ==================================================
# 节点 1: 分析问题（多候选 + KB 判断）
# ==================================================

def analyze_question(state: AgentState) -> dict:
    question = _get_last_user_message(state)

    response = router_llm.invoke([
        SystemMessage(content=ANALYZE_QUESTION_PROMPT),
        HumanMessage(content=question),
    ])

    raw = str(response.content).strip()
    data = _extract_json(raw)

    alternatives: list[str] = []
    needs_kb = True  # 安全默认：不确定时检索一下
    if isinstance(data, dict):
        alts = data.get("alternatives")
        if isinstance(alts, list):
            alternatives = [str(x).strip() for x in alts if str(x).strip()]
        if isinstance(data.get("needs_kb"), bool):
            needs_kb = data["needs_kb"]

    if not alternatives:
        alternatives = [raw[:120] if raw else question]

    primary = alternatives[0]
    print(f"📌 分析（{len(alternatives)} 候选, needs_kb={needs_kb}）: {primary}")

    return {
        "intent": "general_qa",
        "analysis": primary,
        "analysis_alternatives": alternatives,
        "needs_kb": needs_kb,
    }


# ==================================================
# 节点 2: 知识库检索（条件性执行）
# ==================================================

def retrieve_kb(state: AgentState) -> dict:
    """
    根据 state.needs_kb 决定是否真正调用 RAG。
    跳过时把 retrieved_context 置空，generate_answer 会回退到 LLM 自身知识。
    """
    if not state.get("needs_kb", True):
        print("⏭️  跳过 KB 检索（无需查询）")
        return {"retrieved_context": "", "kb_sources": []}

    question = _get_last_user_message(state)
    context, sources = retrieve_with_sources(question)
    print(f"🔎 KB 检索完成: 命中 {len(sources)} 个来源")

    return {"retrieved_context": context, "kb_sources": sources}


# ==================================================
# 节点 3: 生成回答（带风格提示）
# ==================================================

def generate_answer(state: AgentState) -> dict:
    question = _get_last_user_message(state)
    analysis = state.get("analysis", "")
    context = state.get("retrieved_context", "") or "（本次未查询知识库，使用模型自身知识回答）"
    style_key = (state.get("style_hint") or "").strip()
    style_text = STYLE_HINTS.get(style_key, "保持平实清晰的语气。")

    raw_ctx = state.get("retrieved_context", "") or ""
    has_context = bool(raw_ctx) and "未找到" not in raw_ctx and "检索失败" not in raw_ctx
    answer_source = "kb" if has_context else "model"

    prompt = GENERAL_ANSWER_PROMPT.format(
        analysis=analysis,
        context=context,
        style=style_text,
    )
    response = answer_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question),
    ])

    print(f"✅ 回答生成完成 | 风格: {style_key or 'default'} | 依据: {answer_source}")

    return {
        "messages": [AIMessage(content=response.content)],
        "answer_source": answer_source,
    }
