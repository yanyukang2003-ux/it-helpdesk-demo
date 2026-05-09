"""
动态思考流程（plan-and-execute）
================================

不再使用固定的 LangGraph 图。改为：

1. **规划阶段**：用 LLM 看用户问题，决定要走几步、每步做什么。
2. **执行阶段**：按计划逐步执行，每个非 retrieval/answer 步骤同步产出
   2 条备选 primary，方便用户在 UI 上切换思路。
3. **分支重跑**：用户在某一步 override primary 或风格后，从该步往下重跑。

Run state 存在内存里 ``RUNS``（key=thread_id），重启即丢失 —— 与之前
``MemorySaver`` 的行为一致。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import config
from app.rag import retrieve_with_sources, calculate_confidence
from app.prompts import (
    PLANNER_PROMPT,
    THOUGHT_STEP_PROMPT,
    ANSWER_STEP_PROMPT,
    STYLE_HINTS,
)
from app.evaluator import evaluate_and_refine
from app.security import desensitize_rag_output, UserContext, audit_logger
from app.mcp_server import mcp_server
from app.memory import MemoryManager


# --------------------------------------------------
# Data
# --------------------------------------------------

VALID_KINDS = {"analysis", "retrieval", "tool", "reasoning", "summarize", "answer"}


@dataclass
class Step:
    id: str
    title: str
    kind: str
    instruction: str


@dataclass
class StepOutput:
    primary: str
    alternatives: list[str] = field(default_factory=list)
    alt_summaries: list[str] = field(default_factory=list)  # 每条备选的简短特征（≤12字）
    metadata: dict[str, Any] = field(default_factory=dict)
    style_hint: str = ""              # 当前 step 应用的风格（仅 answer 类）
    selected_alt_idx: int = 0         # 当前选中的 primary 来自第几条候选（0=主路径）


@dataclass
class RunState:
    thread_id: str
    user_message: str
    plan: list[Step]
    outputs: list[StepOutput] = field(default_factory=list)


# In-memory store. Cleared on process restart.
RUNS: dict[str, RunState] = {}


# --------------------------------------------------
# LLMs
# --------------------------------------------------

planner_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.3)
exec_llm = ChatOpenAI(model=config.ANSWER_MODEL, temperature=0.4)


# --------------------------------------------------
# Helpers
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


def cp_id(thread_id: str, idx: int) -> str:
    return f"{thread_id}:{idx}"


def parse_cp_id(cp: str) -> tuple[str, int]:
    parts = cp.rsplit(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid checkpoint id: {cp}")
    return parts[0], int(parts[1])


def _format_prior(outputs: list[StepOutput], plan: list[Step], up_to: int) -> str:
    if up_to <= 0:
        return "（无）"
    lines = []
    for i in range(min(up_to, len(outputs))):
        title = plan[i].title if i < len(plan) else f"步骤 {i+1}"
        lines.append(f"[{i+1}] {title}: {outputs[i].primary}")
    return "\n".join(lines) if lines else "（无）"


def _gather_kb_context(outputs: list[StepOutput]) -> str:
    parts = [o.metadata.get("context", "") for o in outputs if o.metadata.get("context")]
    return "\n\n---\n\n".join(p for p in parts if p) or "（本次未引用知识库内容）"


# --------------------------------------------------
# Planner
# --------------------------------------------------

def plan_steps(question: str, memory_context: str = "") -> list[Step]:
    """规划 2~6 步流程。失败时回退到默认 2 步。"""
    # 将记忆上下文拼入用户消息，帮助 Planner 理解对话背景
    if memory_context and memory_context != "（无历史对话）":
        contextualized = f"{question}\n\n【对话历史背景】\n{memory_context}"
    else:
        contextualized = question

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=contextualized),
    ])
    data = _extract_json(str(response.content)) or {}
    raw = data.get("steps") or []

    steps: list[Step] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        kind = str(s.get("kind", "reasoning")).strip()
        if kind not in VALID_KINDS:
            kind = "reasoning"
        steps.append(Step(
            id=f"s{i+1}",
            title=(str(s.get("title", "")).strip() or f"步骤 {i+1}")[:24],
            kind=kind,
            instruction=str(s.get("instruction", "")).strip()[:200],
        ))

    # Cap and ensure terminal answer step
    steps = steps[:6]
    if not steps:
        steps = [
            Step(id="s1", title="理解问题", kind="analysis", instruction="分析用户问题的领域、类型与侧重点"),
            Step(id="s2", title="生成回答", kind="answer", instruction="给出直接清晰的回答"),
        ]
    elif steps[-1].kind != "answer":
        steps.append(Step(id=f"s{len(steps)+1}", title="生成回答",
                          kind="answer", instruction="基于以上结论生成给用户的回答"))
    return steps


# --------------------------------------------------
# Executor
# --------------------------------------------------

def execute_step(
    step: Step,
    plan: list[Step],
    outputs: list[StepOutput],
    question: str,
    idx: int,
    style_hint: str = "",
    user: UserContext | None = None,
    memory_context: str = "",
) -> StepOutput:
    if step.kind == "retrieval":
        return _do_retrieval(step, question)
    if step.kind == "tool":
        return _do_tool(step, question, user)
    if step.kind == "answer":
        return _do_answer(step, plan, outputs, question, style_hint=style_hint, memory_context=memory_context)
    return _do_thought(step, plan, outputs, question, idx)


def _do_thought(step, plan, outputs, question, idx):
    prompt = THOUGHT_STEP_PROMPT.format(
        idx=idx + 1,
        total=len(plan),
        question=question,
        prior=_format_prior(outputs, plan, idx),
        title=step.title,
        kind=step.kind,
        instruction=step.instruction or "（无具体指令）",
    )
    resp = exec_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question),
    ])
    raw = str(resp.content)
    data = _extract_json(raw) or {}
    primary = str(data.get("primary", "")).strip() or raw.strip()[:300]
    alts_raw = data.get("alternatives") or []
    alternatives = [str(a).strip() for a in alts_raw if isinstance(a, (str, int))][:3]
    sums_raw = data.get("alt_summaries") or []
    alt_summaries = [str(s).strip()[:12] for s in sums_raw if isinstance(s, (str, int))][:len(alternatives)]
    return StepOutput(primary=primary, alternatives=alternatives, alt_summaries=alt_summaries)


def _do_retrieval(step, question):
    query = step.instruction or question
    context, sources = retrieve_with_sources(query)
    # 脱敏处理：过滤知识库中可能包含的敏感信息
    context = desensitize_rag_output(context)
    short = []
    for s in sources:
        b = os.path.basename(s) if s else ""
        if b and b not in short:
            short.append(b)
    has = bool(context) and "未找到" not in context and "检索失败" not in context

    # 独立置信度评估
    conf_result = calculate_confidence(query, context, short)
    should_escalate = conf_result.get("should_escalate", False)

    if has:
        primary = (
            f"在知识库中查询「{query[:30]}」，命中 {len(short)} 个文档"
            f"{'（置信度低，建议转人工）' if should_escalate else ''}"
        )
    else:
        primary = f"在知识库中查询「{query[:30]}」，未命中相关文档"

    return StepOutput(
        primary=primary,
        alternatives=[],
        metadata={
            "performed": has and not should_escalate,
            "sources": short,
            "context": context if has else "",
            "query": query,
            "confidence": conf_result,
            "should_escalate": should_escalate,
        },
    )


def _do_tool(step, question, user: UserContext | None = None):
    """执行工具调用步骤。由 Planner 规划，instruction 中含工具名和参数。

    用 LLM 将自然语言指令解析为结构化 MCP 调用参数，
    然后通过 MCP Server 执行，走 RBAC + 审计。
    """
    if user is None:
        user = UserContext(user_id="anonymous")

    # 用 LLM 解析指令 → 结构化参数
    parse_prompt = f"""将以下指令解析为 MCP 工具调用参数。

可用工具：query_ticket(ticket_id)、query_my_tickets(user_id)、create_ticket(title, description, category, priority)

指令: {step.instruction}
用户问题: {question}

严格只输出 JSON，不要任何额外文字：
{{"tool_name": "...", "arguments": {{...}}}}"""

    try:
        resp = planner_llm.invoke(parse_prompt)
        parsed = _extract_json(str(resp.content)) or {}
    except Exception as e:
        return StepOutput(
            primary=f"工具调用解析失败：{e}",
            alternatives=[],
            metadata={"error": str(e)},
        )

    tool_name = parsed.get("tool_name", "")
    tool_args = parsed.get("arguments", {})

    if not tool_name:
        return StepOutput(
            primary="无法从指令中识别工具名称，请检查 Planner 生成的 instruction",
            alternatives=[],
        )

    # 通过 MCP Server 执行（含 RBAC + 审计）
    result = mcp_server.handle_request(
        "tools/call",
        {"name": tool_name, "arguments": tool_args},
        user,
        ""  # session_id 可后续接入
    )

    if "error" in result:
        primary = f"工具 {tool_name} 执行失败：{result['error'].get('message', '未知错误')}"
        audit_logger.log_tool_call(user, tool_name, tool_args, primary, False)
    else:
        content = result.get("content", [])
        primary = content[0].get("text", "") if content else "工具执行完成，无返回内容"
        audit_logger.log_tool_call(user, tool_name, tool_args, primary[:500], True)

    return StepOutput(
        primary=primary,
        alternatives=[],
        metadata={
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": primary,
        },
    )


def _do_answer(step, plan, outputs, question, style_hint: str = "", memory_context: str = ""):
    style_text = STYLE_HINTS.get(style_hint, "保持平实清晰的语气。")
    # 拼接对话历史 + 本轮前序步骤
    full_prior = _format_prior(outputs, plan, len(outputs))
    if memory_context and memory_context != "（无历史对话）":
        full_prior = f"【历史对话背景】\n{memory_context}\n\n【本轮分析】\n{full_prior}"

    prompt = ANSWER_STEP_PROMPT.format(
        question=question,
        prior=full_prior,
        kb_context=_gather_kb_context(outputs),
        style=style_text,
        instruction=step.instruction or "（无具体指令）",
    )
    resp = exec_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question),
    ])
    text = str(resp.content).strip()

    # PGE 反思循环：评估 + 不达标自动重生成
    prior_text = full_prior  # 含记忆上下文 + 本轮前序步骤

    def _regenerate(feedback: str) -> str:
        regen_prompt = ANSWER_STEP_PROMPT.format(
            question=question,
            prior=prior_text,
            kb_context=_gather_kb_context(outputs),
            style=style_text + f"\n\n【改进要求】{feedback}",
            instruction=step.instruction or "（无具体指令）",
        )
        r = exec_llm.invoke([
            SystemMessage(content=regen_prompt),
            HumanMessage(content=question),
        ])
        return str(r.content).strip()

    refined, eval_result, regen_count = evaluate_and_refine(
        question=question,
        answer=text,
        prior=prior_text,
        generate_fn=_regenerate,
        threshold=config.EVALUATION_THRESHOLD,
    )

    return StepOutput(
        primary=refined,
        alternatives=[],
        style_hint=style_hint,
        metadata={
            "evaluation": eval_result.to_dict(),
            "regenerations": regen_count,
        },
    )


# --------------------------------------------------
# Run / branch
# --------------------------------------------------

def stream_run(thread_id: str, user_message: str, user: UserContext | None = None) -> Iterator[dict]:
    """完整执行：先 plan，再逐步 execute；产出 SSE event 字典。"""
    if user is None:
        user = UserContext(user_id="anonymous")

    # 加载滚动分层压缩记忆
    memory = MemoryManager(thread_id)
    memory_context = memory.get_context_for_llm()

    # 立即发送启动事件，避免 Render proxy 在 OpenAI 调用期间断开连接
    yield {"type": "run_started", "thread_id": thread_id}
    plan = plan_steps(user_message, memory_context)
    state = RunState(thread_id=thread_id, user_message=user_message, plan=plan)
    RUNS[thread_id] = state

    yield {
        "type": "plan_complete",
        "thread_id": thread_id,
        "total_steps": len(plan),
    }

    for i, step in enumerate(plan):
        out = execute_step(step, plan, state.outputs, user_message, i, user=user, memory_context=memory_context)
        state.outputs.append(out)
        yield {
            "type": "step_complete",
            "index": i,
            "step": asdict(step),
            "output": asdict(out),
            "checkpoint_id": cp_id(thread_id, i),
        }

    final = state.outputs[-1].primary if state.outputs else ""

    # 持久化本轮对话到记忆系统
    memory.add_turn("user", user_message)
    memory.add_turn("ai", final[:2000])

    yield {
        "type": "done",
        "reply": final,
        "thread_id": thread_id,
    }


def stream_branch(
    thread_id: str,
    from_index: int,
    *,
    override_primary: str | None = None,
    override_alt_idx: int | None = None,
    style_hint: str | None = None,
    override_user_message: str | None = None,
    as_new_thread: bool = False,
) -> Iterator[dict]:
    """从某步分叉重跑。yield 事件类似 stream_run 中的 step_complete。"""
    src = RUNS.get(thread_id)
    if src is None:
        raise ValueError("不存在的会话")

    # Clone if forking
    if as_new_thread:
        new_tid = str(uuid.uuid4())
        cloned = RunState(
            thread_id=new_tid,
            user_message=src.user_message,
            plan=[Step(**asdict(s)) for s in src.plan],
            outputs=[StepOutput(**asdict(o)) for o in src.outputs],
        )
        RUNS[new_tid] = cloned
        src = cloned
        thread_id = new_tid
        yield {"type": "thread_forked", "thread_id": new_tid}

    # Re-plan if user message changed
    if override_user_message is not None:
        src.user_message = override_user_message
        src.plan = plan_steps(override_user_message)
        src.outputs = []
        yield {
            "type": "plan_complete",
            "thread_id": src.thread_id,
            "total_steps": len(src.plan),
        }
        for i, step in enumerate(src.plan):
            out = execute_step(step, src.plan, src.outputs, src.user_message, i)
            src.outputs.append(out)
            yield {
                "type": "step_complete",
                "index": i,
                "step": asdict(step),
                "output": asdict(out),
                "checkpoint_id": cp_id(src.thread_id, i),
            }
        yield {
            "type": "done",
            "reply": src.outputs[-1].primary if src.outputs else "",
            "thread_id": src.thread_id,
        }
        return

    if from_index < 0 or from_index >= len(src.plan):
        raise ValueError(f"无效的步骤索引 {from_index}")

    # In-place override of step k's primary.
    if override_primary is not None:
        if from_index < len(src.outputs):
            src.outputs[from_index].primary = override_primary
            if override_alt_idx is not None:
                src.outputs[from_index].selected_alt_idx = override_alt_idx
        else:
            src.outputs.append(StepOutput(primary=override_primary))
        # Truncate downstream
        src.outputs = src.outputs[: from_index + 1]
        yield {
            "type": "step_complete",
            "index": from_index,
            "step": asdict(src.plan[from_index]),
            "output": asdict(src.outputs[from_index]),
            "checkpoint_id": cp_id(src.thread_id, from_index),
        }
        start = from_index + 1
    elif style_hint is not None:
        # Re-execute step k with new style (only valid for answer kind)
        if src.plan[from_index].kind != "answer":
            raise ValueError("style_hint 只对 answer 类型步骤有效")
        # Truncate including k, then re-run
        src.outputs = src.outputs[:from_index]
        new_out = execute_step(src.plan[from_index], src.plan, src.outputs,
                                src.user_message, from_index, style_hint=style_hint)
        src.outputs.append(new_out)
        yield {
            "type": "step_complete",
            "index": from_index,
            "step": asdict(src.plan[from_index]),
            "output": asdict(new_out),
            "checkpoint_id": cp_id(src.thread_id, from_index),
        }
        start = from_index + 1
    else:
        # Just re-run downstream from from_index+1 unchanged
        src.outputs = src.outputs[: from_index + 1]
        start = from_index + 1

    # Run remaining steps
    for i in range(start, len(src.plan)):
        out = execute_step(src.plan[i], src.plan, src.outputs, src.user_message, i)
        src.outputs.append(out)
        yield {
            "type": "step_complete",
            "index": i,
            "step": asdict(src.plan[i]),
            "output": asdict(out),
            "checkpoint_id": cp_id(src.thread_id, i),
        }

    yield {
        "type": "done",
        "reply": src.outputs[-1].primary if src.outputs else "",
        "thread_id": src.thread_id,
    }


# --------------------------------------------------
# Read APIs (for /threads/{tid}/checkpoints replacement)
# --------------------------------------------------

def list_run_steps(thread_id: str) -> list[dict]:
    src = RUNS.get(thread_id)
    if src is None:
        return []
    rows = []
    for i, step in enumerate(src.plan):
        rows.append({
            "checkpoint_id": cp_id(thread_id, i),
            "step_index": i,
            "step": asdict(step),
            "output": asdict(src.outputs[i]) if i < len(src.outputs) else None,
        })
    return rows
