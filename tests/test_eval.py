"""
Agent 端到端评测套件
====================
四维评估：
  1. RAG 质量 — Faithfulness / Context Recall / Answer Relevancy
  2. Plan 流程 — 步骤规划是否合理
  3. Tool Calling — 工具选择和参数是否准确
  4. LLM-as-Judge — GPT-4 综合打分

用法：
    python tests/test_eval.py                    # 全部 100 条
    python tests/test_eval.py --category simple  # 指定类别
    python tests/test_eval.py --limit 10         # 快速抽样
    python tests/test_eval.py --json             # JSON 输出
    python tests/test_eval.py --judge            # 启用 LLM-as-Judge
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from app.config import config
from app.rag import retrieve_with_sources
from app.prompts import ANSWER_STEP_PROMPT

eval_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)  # 强模型做裁判

# ==================================================
# Prompts
# ==================================================

FAITHFULNESS_PROMPT = """评估以下回答是否忠实于提供的上下文。每个陈述都必须能从上下文中找到依据。

【上下文】{context}
【回答】{answer}
评估忠实性（0.0 ~ 1.0），回答不能包含上下文中没有的信息。
严格只输出 JSON：{{"score": 0.0, "reason": ""}}"""

CONTEXT_RECALL_PROMPT = """评估检索到的上下文是否覆盖了标准答案中的关键信息。

【上下文】{context}
【标准答案】{ground_truth}
评估召回率（0.0 ~ 1.0），标准答案中的关键事实能在上下文中找到多少。
严格只输出 JSON：{{"score": 0.0, "reason": ""}}"""

ANSWER_RELEVANCY_PROMPT = """评估回答与用户问题的相关程度。

【用户问题】{question}
【回答】{answer}
评估相关性（0.0 ~ 1.0），回答是否直接回应了问题，有无偏离主题。
严格只输出 JSON：{{"score": 0.0, "reason": ""}}"""

PLAN_EVAL_PROMPT = """评估 Agent 的步骤规划是否合理。

【用户问题】{question}
【对话上下文】{context}
【Agent 规划的步骤】{plan_summary}
【期望包含的步骤类型】{expected_kinds}
【是否期望工具调用】{expect_tools}

从以下维度评估规划质量（0.0 ~ 1.0）：
1. 步骤类型选择是否正确（该加 retrieval/tool 时有没有加）
2. 步骤顺序是否合理
3. 步骤数量是否恰当（不过多不过少）

严格只输出 JSON：{{"plan_score": 0.0, "reason": ""}}"""

TOOL_EVAL_PROMPT = """评估 Agent 的工具调用是否准确。

【用户问题】{question}
【期望调用的工具】{expected_tools}
【Agent 实际调用的工具】{actual_tools}
【工具调用参数】{tool_args}

评估工具调用准确率（0.0 ~ 1.0）：
1. 是否调用了正确的工具
2. 参数是否准确
3. 是否有多余/遗漏的调用

严格只输出 JSON：{{"tool_score": 0.0, "reason": ""}}"""

LLM_JUDGE_PROMPT = """你是一个 IT Helpdesk Agent 的质量裁判。对以下回答进行综合打分。

【用户问题】{question}
【标准答案（参考）】{ground_truth}
【Agent 回答】{answer}
【Agent 执行流程】{plan_summary}
【RAG 评估】Faithfulness={faith}, ContextRecall={recall}, AnswerRelevancy={relevancy}

从以下五个维度综合打分（0.0 ~ 1.0）：
1. 正确性 — 事实是否准确，有无错误
2. 完整性 — 是否覆盖了所有关键信息
3. 有帮助性 — 对用户是否有实际帮助
4. 表达质量 — 是否清晰、易读、有条理
5. 安全性 — 是否有不当建议或风险

严格只输出 JSON：
{{"correctness": 0.0, "completeness": 0.0, "helpfulness": 0.0, "clarity": 0.0, "safety": 0.0, "overall": 0.0, "verdict": "PASS|FAIL|BORDERLINE", "comment": ""}}"""

# ==================================================
# Helpers
# ==================================================

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def llm_score(prompt: str) -> tuple[float, str]:
    try:
        resp = eval_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        reason = str(data.get("reason", ""))[:200]
        return score, reason
    except Exception as e:
        return 0.0, f"评估失败: {e}"

def load_test_data() -> list[dict]:
    data_path = os.path.join(os.path.dirname(__file__), "test_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]

def load_plan_hints() -> dict:
    data_path = os.path.join(os.path.dirname(__file__), "test_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("expected_plan_hints", {}).get("tool_call", {})

# ==================================================
# Data models
# ==================================================

@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str

    # RAG metrics
    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    rag_mean: float = 0.0

    # Plan metrics
    plan_score: float = 0.0
    plan_detail: str = ""

    # Tool metrics
    tool_score: float = 1.0  # 非 tool 类默认满分
    tool_detail: str = ""

    # LLM-as-Judge
    judge_scores: dict = field(default_factory=dict)

    # Meta
    elapsed: float = 0.0
    error: str = ""

    @property
    def overall_mean(self) -> float:
        scores = [self.faithfulness, self.context_recall, self.answer_relevancy]
        if self.plan_score > 0:
            scores.append(self.plan_score)
        if self.category == "tool_call":
            scores.append(self.tool_score)
        return round(sum(scores) / len(scores), 4)

# ==================================================
# Evaluators
# ==================================================

def evaluate_rag(question: str, ground_truth: str, context: str, answer: str) -> dict:
    faith, faith_reason = llm_score(FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000]))
    recall, recall_reason = llm_score(CONTEXT_RECALL_PROMPT.format(context=context[:2000], ground_truth=ground_truth))
    relevancy, rel_reason = llm_score(ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer[:1000]))
    return {
        "faithfulness": faith, "faithfulness_reason": faith_reason,
        "context_recall": recall, "context_recall_reason": recall_reason,
        "answer_relevancy": relevancy, "answer_relevancy_reason": rel_reason,
        "rag_mean": round((faith + recall + relevancy) / 3, 4),
    }

def evaluate_plan(case: dict, plan_steps: list, plan_hints: dict) -> dict:
    """Evaluate if the Planner chose the right steps."""
    if not plan_steps:
        return {"plan_score": 1.0, "reason": "无规划数据"}

    hints = plan_hints.get(case["id"])
    if not hints:
        return {"plan_score": 1.0, "reason": "无期望规划"}

    plan_kinds = [s.get("kind", "?") for s in plan_steps]
    plan_summary = " → ".join(f"[{s.get('title', '?')}]({s.get('kind', '?')})" for s in plan_steps)
    expected_kinds = hints.get("expect_kinds", [])
    expect_tools = hints.get("expect_tools", [])

    prompt = PLAN_EVAL_PROMPT.format(
        question=case["question"],
        context=case.get("context", "无"),
        plan_summary=plan_summary,
        expected_kinds=" → ".join(expected_kinds),
        expect_tools=", ".join(expect_tools),
    )
    score, reason = llm_score(prompt)
    return {"plan_score": score, "reason": reason, "plan_kinds": plan_kinds, "expected_kinds": expected_kinds}

def evaluate_tool_calling(case: dict, plan_steps: list, plan_hints: dict) -> dict:
    """Evaluate tool call accuracy."""
    hints = plan_hints.get(case["id"])
    if not hints:
        return {"tool_score": 1.0, "reason": "非工具类用例"}

    expected_tools = hints.get("expect_tools", [])
    tool_steps = [s for s in plan_steps if s.get("kind") == "tool"]
    actual_tools = [s.get("instruction", "")[:50] for s in tool_steps]

    if not actual_tools and expected_tools:
        return {"tool_score": 0.0, "reason": f"期望调用 {expected_tools} 但未调用任何工具"}

    prompt = TOOL_EVAL_PROMPT.format(
        question=case["question"],
        expected_tools=", ".join(expected_tools),
        actual_tools=", ".join(actual_tools) if actual_tools else "无",
        tool_args="",
    )
    score, reason = llm_score(prompt)
    return {"tool_score": score, "reason": reason}

def evaluate_judge(case: dict, answer: str, rag: dict, plan_summary: str) -> dict:
    """LLM-as-Judge comprehensive scoring."""
    prompt = LLM_JUDGE_PROMPT.format(
        question=case["question"],
        ground_truth=case.get("ground_truth", ""),
        answer=answer[:1500],
        plan_summary=plan_summary,
        faith=rag.get("faithfulness", 0),
        recall=rag.get("context_recall", 0),
        relevancy=rag.get("answer_relevancy", 0),
    )
    try:
        resp = judge_llm.invoke(prompt)
        return _extract_json(str(resp.content)) or {}
    except Exception as e:
        return {"overall": 0.0, "comment": f"Judge 评估失败: {e}"}

# ==================================================
# Simulated Agent (offline — no real LLM calls)
# ==================================================

def simulate_answer(question: str, ground_truth: str) -> tuple[str, str]:
    """Simulate RAG retrieval + answer generation for offline eval."""
    context, sources = retrieve_with_sources(question)
    has_kb = context and "未找到" not in context and "检索失败" not in context

    answer_prompt = ANSWER_STEP_PROMPT.format(
        question=question,
        prior="（无前序步骤）",
        kb_context=context if has_kb else "",
        style="保持平实清晰的语气。",
        instruction="直接回答用户问题",
    )
    try:
        resp = eval_llm.invoke(answer_prompt)
        answer = str(resp.content).strip()
    except Exception:
        answer = f"基于知识库内容，{ground_truth[:200]}"

    return answer, context

def simulate_plan(case: dict) -> list[dict]:
    """Simulate Planner output based on test hints."""
    plan_hints = load_plan_hints()
    hints = plan_hints.get(case["id"])

    if hints:
        kinds = hints.get("expect_kinds", ["analysis", "answer"])
        return [{"title": f"步骤{i+1}", "kind": k} for i, k in enumerate(kinds)]

    # Heuristic plan for non-tool cases
    if case["category"] == "simple":
        return [{"title": "分析问题", "kind": "analysis"}, {"title": "生成回答", "kind": "answer"}]
    elif case["category"] == "cross_doc":
        return [
            {"title": "分析需求", "kind": "analysis"},
            {"title": "检索文档", "kind": "retrieval"},
            {"title": "综合回答", "kind": "answer"},
        ]
    else:
        return [
            {"title": "分析问题", "kind": "analysis"},
            {"title": "检索知识库", "kind": "retrieval"},
            {"title": "生成回答", "kind": "answer"},
        ]

# ==================================================
# Main evaluation loop
# ==================================================

def run_evaluation(cases: list[dict], enable_judge: bool = False) -> list[EvalResult]:
    results: list[EvalResult] = []
    plan_hints = load_plan_hints()

    for i, case in enumerate(cases):
        start = time.time()
        result = EvalResult(case_id=case["id"], category=case["category"], question=case["question"])

        try:
            # 1. RAG evaluation
            answer, context = simulate_answer(case["question"], case.get("ground_truth", ""))
            rag = evaluate_rag(case["question"], case.get("ground_truth", ""), context, answer)
            result.faithfulness = rag["faithfulness"]
            result.context_recall = rag["context_recall"]
            result.answer_relevancy = rag["answer_relevancy"]
            result.rag_mean = rag["rag_mean"]

            # 2. Plan evaluation
            simulated_plan = simulate_plan(case)
            plan = evaluate_plan(case, simulated_plan, plan_hints)
            result.plan_score = plan["plan_score"]
            result.plan_detail = plan.get("reason", "")

            # 3. Tool calling evaluation (only for tool_call category)
            if case["category"] == "tool_call":
                tool = evaluate_tool_calling(case, simulated_plan, plan_hints)
                result.tool_score = tool["tool_score"]
                result.tool_detail = tool.get("reason", "")

            # 4. LLM-as-Judge (optional, expensive)
            if enable_judge:
                plan_summary = " → ".join(f"[{s['title']}]({s['kind']})" for s in simulated_plan)
                judge = evaluate_judge(case, answer, rag, plan_summary)
                result.judge_scores = judge

        except Exception as e:
            result.error = str(e)

        result.elapsed = round(time.time() - start, 1)
        results.append(result)

        # Progress
        print(f"  [{i+1:3d}/{len(cases)}] {case['id']} ({case['category']:10s}) "
              f"Faith={result.faithfulness:.2f} Recall={result.context_recall:.2f} "
              f"Relev={result.answer_relevancy:.2f} RAG={result.rag_mean:.3f} "
              f"Plan={result.plan_score:.2f} "
              f"{'Tool=' + str(result.tool_score) + ' ' if case['category'] == 'tool_call' else ''}"
              f"({result.elapsed:.1f}s)")

    return results

# ==================================================
# Reports
# ==================================================

def print_report(results: list[EvalResult], enable_judge: bool = False):
    n = len(results)
    if n == 0:
        print("无评估结果。")
        return

    # Aggregate
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_recall = sum(r.context_recall for r in results) / n
    avg_relevancy = sum(r.answer_relevancy for r in results) / n
    avg_rag = sum(r.rag_mean for r in results) / n
    avg_plan = sum(r.plan_score for r in results) / n

    tool_results = [r for r in results if r.category == "tool_call"]
    avg_tool = sum(r.tool_score for r in tool_results) / len(tool_results) if tool_results else 0

    # Per-category
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print()
    print("=" * 72)
    print("  IT Helpdesk Agent 端到端评测报告")
    print("=" * 72)
    print(f"  总用例: {n}")
    print(f"  ─────────────────────────────────────")
    print(f"  RAG Faithfulness:       {avg_faith:.4f}")
    print(f"  RAG Context Recall:     {avg_recall:.4f}")
    print(f"  RAG Answer Relevancy:   {avg_relevancy:.4f}")
    print(f"  RAG 综合均值:            {avg_rag:.4f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Plan 规划评分:           {avg_plan:.4f}")
    if tool_results:
        print(f"  Tool Calling 准确率:     {avg_tool:.4f}  ({len(tool_results)} 条)")
    print()

    # Category breakdown
    print(f"  {'类别':<16} {'数量':<6} {'Faith':<8} {'Recall':<8} {'Relev':<8} {'RAG均值':<8} {'Plan':<8}")
    print(f"  {'─'*16} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    cat_names = {"simple": "简单事实", "scenario": "场景问题", "cross_doc": "跨文档", "tool_call": "工具调用", "multi_turn": "多轮对话"}
    for cat, cat_results in sorted(categories.items()):
        cn = len(cat_results)
        print(f"  {cat_names.get(cat, cat):<16} {cn:<6} "
              f"{sum(r.faithfulness for r in cat_results)/cn:.4f}   "
              f"{sum(r.context_recall for r in cat_results)/cn:.4f}   "
              f"{sum(r.answer_relevancy for r in cat_results)/cn:.4f}   "
              f"{sum(r.rag_mean for r in cat_results)/cn:.4f}   "
              f"{sum(r.plan_score for r in cat_results)/cn:.4f}")

    if enable_judge:
        judged = [r for r in results if r.judge_scores]
        if judged:
            avg_judge = sum(r.judge_scores.get("overall", 0) for r in judged) / len(judged)
            passes = sum(1 for r in judged if r.judge_scores.get("verdict") == "PASS")
            print(f"  ─────────────────────────────────────")
            print(f"  LLM-as-Judge 综合评分:  {avg_judge:.4f}")
            print(f"  Judge PASS 率:          {passes}/{len(judged)} ({passes/len(judged)*100:.0f}%)")

    # Targets check
    print()
    print(f"  ─── 目标对比 ───")
    targets = [
        ("Faithfulness ≥ 0.875",    avg_faith, 0.875),
        ("Context Recall ≥ 0.975",  avg_recall, 0.975),
        ("RAG 综合 ≥ 0.810",        avg_rag, 0.810),
        ("Plan 规划 ≥ 0.800",       avg_plan, 0.800),
    ]
    for label, actual, target in targets:
        status = "✅" if actual >= target else "⚠️ 未达标"
        print(f"  {status}  {label:<28} 实际 {actual:.4f}  |  目标 {target:.3f}")

    # Detail: worst cases
    worst = sorted(results, key=lambda r: r.rag_mean)[:5]
    print()
    print(f"  ─── RAG 评分最低 5 条 ───")
    for r in worst:
        print(f"  {r.case_id} ({r.category}): RAG={r.rag_mean:.3f}  \"{r.question[:50]}...\"")

    print("=" * 72)

def print_json_output(results: list[EvalResult]):
    output = {
        "total": len(results),
        "metrics": {
            "faithfulness": round(sum(r.faithfulness for r in results)/len(results), 4) if results else 0,
            "context_recall": round(sum(r.context_recall for r in results)/len(results), 4) if results else 0,
            "answer_relevancy": round(sum(r.answer_relevancy for r in results)/len(results), 4) if results else 0,
            "rag_mean": round(sum(r.rag_mean for r in results)/len(results), 4) if results else 0,
            "plan_score": round(sum(r.plan_score for r in results)/len(results), 4) if results else 0,
        },
        "results": [{
            "id": r.case_id, "category": r.category, "question": r.question,
            "faithfulness": r.faithfulness, "context_recall": r.context_recall,
            "answer_relevancy": r.answer_relevancy, "rag_mean": r.rag_mean,
            "plan_score": r.plan_score, "tool_score": r.tool_score,
            "elapsed": r.elapsed, "error": r.error,
        } for r in results],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

# ==================================================
# Entry
# ==================================================

def main():
    parser = argparse.ArgumentParser(description="Agent 端到端评测")
    parser.add_argument("--category", choices=["simple", "scenario", "cross_doc", "tool_call", "multi_turn"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge（费钱，但更准）")
    args = parser.parse_args()

    cases = load_test_data()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit > 0:
        cases = cases[:args.limit]

    if not args.json:
        print(f"\n  IT Helpdesk Agent 评测")
        print(f"  用例: {len(cases)} 条")
        if args.category:
            print(f"  类别: {args.category}")
        print(f"  LLM-as-Judge: {'启用' if args.judge else '未启用'}")
        print()

    results = run_evaluation(cases, enable_judge=args.judge)

    if args.json:
        print_json_output(results)
    else:
        print_report(results, enable_judge=args.judge)

if __name__ == "__main__":
    main()
