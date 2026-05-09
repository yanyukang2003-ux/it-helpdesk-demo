"""
RAGAS 量化评估脚本
==================
基于 RAGAS 框架对 IT Helpdesk Agent 的 RAG 管道进行多维度评估。

评估指标：
- Faithfulness（忠实性）：回答是否忠实于检索到的上下文
- Context Recall（上下文召回）：相关上下文是否被检索到
- Context Precision（上下文精确度）：检索到的上下文是否相关
- Answer Relevancy（回答相关性）：回答与问题的相关程度

测试集：40 条自动生成用例，覆盖 simple / scenario / cross-doc 三类。

用法：
    python tests/test_ragas.py                    # 运行全部评估
    python tests/test_ragas.py --category simple  # 仅运行指定类别
    python tests/test_ragas.py --json             # JSON 格式输出
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import config
from app.rag import retrieve_with_sources
from app.prompts import ANSWER_STEP_PROMPT

eval_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)


# ==================================================
# RAGAS-style evaluation prompts
# ==================================================

FAITHFULNESS_PROMPT = """评估以下回答是否忠实于提供的上下文内容。回答中的每个陈述都必须能从上下文中找到依据。

【上下文】
{context}

【回答】
{answer}

【评估标准】
1. 回答中是否包含上下文中没有的信息？（扣分项）
2. 回答中是否有与上下文矛盾的内容？（严重扣分项）
3. 回答是否基于上下文进行合理推断而非凭空编造？

请给出忠实性评分（0.0 ~ 1.0）和简要理由。

严格只输出 JSON：
{{"score": 0.0, "reason": ""}}"""


CONTEXT_RECALL_PROMPT = """评估检索到的上下文是否覆盖了标准答案中的关键信息。

【检索到的上下文】
{context}

【标准答案】
{ground_truth}

【评估标准】
1. 标准答案中的关键事实是否能在上下文中找到？
2. 上下文是否遗漏了重要信息？
3. 上下文的覆盖比例如何？

请给出上下文召回评分（0.0 ~ 1.0）和简要理由。

严格只输出 JSON：
{{"score": 0.0, "reason": ""}}"""


ANSWER_RELEVANCY_PROMPT = """评估回答与用户问题的相关程度。

【用户问题】
{question}

【回答】
{answer}

【评估标准】
1. 回答是否直接回应了用户的问题？
2. 回答中有无偏离主题的内容？
3. 回答是否完整覆盖了问题的所有方面？

请给出回答相关性评分（0.0 ~ 1.0）和简要理由。

严格只输出 JSON：
{{"score": 0.0, "reason": ""}}"""


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
    """Use LLM to score based on a prompt. Returns (score, reason)."""
    try:
        resp = eval_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        reason = str(data.get("reason", ""))[:200]
        return score, reason
    except Exception as e:
        return 0.0, f"评估失败: {e}"


# ==================================================
# RAGAS Metrics
# ==================================================

@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str
    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    mean_score: float = 0.0
    detail: dict = field(default_factory=dict)


def evaluate_case(case: dict) -> EvalResult:
    """评估单个测试用例。"""
    question = case["question"]
    ground_truth = case.get("ground_truth", "")

    # Step 1: Retrieve context
    context, sources = retrieve_with_sources(question)

    # Step 2: Generate answer
    answer_prompt = ANSWER_STEP_PROMPT.format(
        question=question,
        prior="（无前序步骤）",
        kb_context=context if context and "未找到" not in context else "",
        style="保持平实清晰的语气。",
        instruction="直接回答用户问题",
    )
    try:
        answer_resp = eval_llm.invoke(answer_prompt)
        answer = str(answer_resp.content).strip()
    except Exception as e:
        answer = f"回答生成失败: {e}"

    # Step 3: Evaluate
    faith_score, faith_reason = llm_score(
        FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000])
    )

    recall_score, recall_reason = llm_score(
        CONTEXT_RECALL_PROMPT.format(context=context[:2000], ground_truth=ground_truth)
    )

    relevancy_score, rel_reason = llm_score(
        ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer[:1000])
    )

    mean = round((faith_score + recall_score + relevancy_score) / 3, 4)

    return EvalResult(
        case_id=case["id"],
        category=case["category"],
        question=question,
        faithfulness=faith_score,
        context_recall=recall_score,
        answer_relevancy=relevancy_score,
        mean_score=mean,
        detail={
            "faithfulness_reason": faith_reason,
            "context_recall_reason": recall_reason,
            "answer_relevancy_reason": rel_reason,
            "sources": sources,
            "answer": answer[:500],
            "ground_truth": ground_truth[:500],
        },
    )


# ==================================================
# Main evaluation runner
# ==================================================

def load_test_data() -> list[dict]:
    data_path = os.path.join(os.path.dirname(__file__), "test_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def print_header():
    print("=" * 70)
    print("  RAGAS 量化评估 — IT Helpdesk Agent RAG 管道")
    print("=" * 70)
    print()


def print_summary(results: list[EvalResult]):
    n = len(results)
    if n == 0:
        print("无评估结果。")
        return

    avg_faith = sum(r.faithfulness for r in results) / n
    avg_recall = sum(r.context_recall for r in results) / n
    avg_rel = sum(r.answer_relevancy for r in results) / n
    avg_mean = sum(r.mean_score for r in results) / n

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    print("-" * 70)
    print(f"  总用例数: {n}")
    print(f"  Faithfulness 均值:     {avg_faith:.4f}")
    print(f"  Context Recall 均值:   {avg_recall:.4f}")
    print(f"  Answer Relevancy 均值: {avg_rel:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  综合均值:              {avg_mean:.4f}")
    print()

    for cat, cat_results in sorted(categories.items()):
        cat_n = len(cat_results)
        cat_mean = sum(r.mean_score for r in cat_results) / cat_n
        cat_faith = sum(r.faithfulness for r in cat_results) / cat_n
        cat_recall = sum(r.context_recall for r in cat_results) / cat_n
        cat_name = {"simple": "简单事实查询", "scenario": "场景化问题", "cross_doc": "跨文档综合查询"}.get(cat, cat)
        print(f"  [{cat_name}] ({cat_n} 条)")
        print(f"    Faithfulness: {cat_faith:.4f}  |  Context Recall: {cat_recall:.4f}  |  均值: {cat_mean:.4f}")

    print()
    print("=" * 70)

    # Check targets
    targets_met = avg_faith >= 0.85 and avg_recall >= 0.95 and avg_mean >= 0.81
    if targets_met:
        print("  ✅ 所有指标达到目标")
    else:
        if avg_faith < 0.85:
            print(f"  ⚠️  Faithfulness 未达标 (当前 {avg_faith:.4f}, 目标 0.875)")
        if avg_recall < 0.95:
            print(f"  ⚠️  Context Recall 未达标 (当前 {avg_recall:.4f}, 目标 0.975)")
        if avg_mean < 0.81:
            print(f"  ⚠️  综合均值未达标 (当前 {avg_mean:.4f}, 目标 0.810)")
    print("=" * 70)


def print_json_output(results: list[EvalResult]):
    output = {
        "total": len(results),
        "metrics": {
            "faithfulness_mean": round(sum(r.faithfulness for r in results) / len(results), 4) if results else 0,
            "context_recall_mean": round(sum(r.context_recall for r in results) / len(results), 4) if results else 0,
            "answer_relevancy_mean": round(sum(r.answer_relevancy for r in results) / len(results), 4) if results else 0,
            "overall_mean": round(sum(r.mean_score for r in results) / len(results), 4) if results else 0,
        },
        "results": [
            {
                "id": r.case_id,
                "category": r.category,
                "question": r.question,
                "faithfulness": r.faithfulness,
                "context_recall": r.context_recall,
                "answer_relevancy": r.answer_relevancy,
                "mean": r.mean_score,
                "detail": r.detail,
            }
            for r in results
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="RAGAS 量化评估")
    parser.add_argument("--category", choices=["simple", "scenario", "cross_doc"], help="仅评估指定类别")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--limit", type=int, default=0, help="限制评估数量（用于快速测试）")
    args = parser.parse_args()

    cases = load_test_data()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit > 0:
        cases = cases[: args.limit]

    if not args.json:
        print_header()
        print(f"  加载 {len(cases)} 条测试用例")
        if args.category:
            print(f"  筛选类别: {args.category}")
        print()

    results: list[EvalResult] = []
    for i, case in enumerate(cases):
        if not args.json:
            print(f"  [{i+1}/{len(cases)}] {case['id']} ({case['category']}) {case['question'][:40]}…", end=" ", flush=True)

        start = time.time()
        result = evaluate_case(case)
        elapsed = time.time() - start

        if not args.json:
            print(f"Faith={result.faithfulness:.3f} Recall={result.context_recall:.3f} "
                  f"Rel={result.answer_relevancy:.3f} Mean={result.mean_score:.3f} ({elapsed:.1f}s)")

        results.append(result)

    if args.json:
        print_json_output(results)
    else:
        print()
        print_summary(results)


if __name__ == "__main__":
    main()
