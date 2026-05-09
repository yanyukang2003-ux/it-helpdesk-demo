"""
PGE 反思循环（Plan-Generate-Evaluate）
======================================

Evaluator 对 answer 步骤的产出从四个维度打分：
  - relevance（相关性）
  - faithfulness（忠实性）
  - completeness（完整性）
  - conciseness（简洁性）

综合得分 < 阈值时，携带结构化 feedback 触发 Generator 重生成。
最多重试 MAX_REGENERATIONS 次，超出后取最高分版本。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import config
from app.prompts import EVALUATOR_PROMPT

evaluator_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)


@dataclass
class EvaluationResult:
    relevance: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    overall: float = 0.0
    feedback: str = ""
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "conciseness": self.conciseness,
            "overall": self.overall,
            "feedback": self.feedback,
            "passed": self.passed,
        }


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


def evaluate_answer(
    question: str,
    answer: str,
    prior: str = "",
    threshold: float | None = None,
) -> EvaluationResult:
    """对回答质量进行四维度评估。

    Args:
        question: 用户原始问题
        answer: 生成的回答
        prior: 前序步骤产出摘要
        threshold: 通过阈值，默认使用 config.CONFIDENCE_THRESHOLD

    Returns:
        EvaluationResult 包含各维度分数、综合分、feedback 和是否通过
    """
    threshold = threshold if threshold is not None else config.EVALUATION_THRESHOLD

    prompt = EVALUATOR_PROMPT.format(
        question=question,
        prior=prior or "（无前序步骤）",
        answer=answer,
    )

    try:
        resp = evaluator_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
    except Exception as e:
        print(f"⚠️ Evaluator 调用失败: {e}")
        return EvaluationResult(overall=1.0, passed=True)

    relevance = max(0.0, min(1.0, float(data.get("relevance", 0.7))))
    faithfulness = max(0.0, min(1.0, float(data.get("faithfulness", 0.7))))
    completeness = max(0.0, min(1.0, float(data.get("completeness", 0.7))))
    conciseness = max(0.0, min(1.0, float(data.get("conciseness", 0.7))))

    # Weighted overall score
    overall = round(
        relevance * 0.30 + faithfulness * 0.30 + completeness * 0.25 + conciseness * 0.15,
        4,
    )

    feedback = str(data.get("feedback", "")).strip()
    passed = overall >= threshold

    return EvaluationResult(
        relevance=relevance,
        faithfulness=faithfulness,
        completeness=completeness,
        conciseness=conciseness,
        overall=overall,
        feedback=feedback,
        passed=passed,
    )


MAX_REGENERATIONS = 3


def evaluate_and_refine(
    question: str,
    answer: str,
    prior: str = "",
    generate_fn: callable = None,
    threshold: float | None = None,
) -> tuple[str, EvaluationResult, int]:
    """评估回答并在不达标时自动重生成。

    Args:
        question: 用户原始问题
        answer: 初始回答
        prior: 前序步骤产出
        generate_fn: 重生成函数，接受 (feedback: str) -> str
        threshold: 通过阈值

    Returns:
        (final_answer, best_evaluation, regeneration_count)
    """
    threshold = threshold if threshold is not None else config.EVALUATION_THRESHOLD

    best_answer = answer
    best_eval = evaluate_answer(question, answer, prior, threshold)
    regeneration_count = 0

    while not best_eval.passed and regeneration_count < MAX_REGENERATIONS:
        if generate_fn is None:
            break

        regeneration_count += 1
        print(f"🔄 PGE 反思循环 — 第 {regeneration_count} 次重生成 (当前得分: {best_eval.overall})")

        try:
            new_answer = generate_fn(best_eval.feedback)
        except Exception as e:
            print(f"⚠️ 重生成失败: {e}")
            break

        new_eval = evaluate_answer(question, new_answer, prior, threshold)

        if new_eval.overall > best_eval.overall:
            best_answer = new_answer
            best_eval = new_eval

        if new_eval.passed:
            break

    if best_eval.passed:
        print(f"✅ PGE 评估通过 (得分: {best_eval.overall}, 重生成 {regeneration_count} 次)")
    else:
        print(f"⚠️ PGE 评估未达阈值 (最高分: {best_eval.overall}, 重生成 {regeneration_count} 次)")

    return best_answer, best_eval, regeneration_count
