"""
Agent 状态定义
LangGraph 的核心概念：所有节点共享并更新同一个状态对象
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 的全局状态"""

    # 对话消息历史
    # add_messages 注解让 LangGraph 自动追加新消息，而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]

    # 意图分类结果
    # 可选值: knowledge_qa / ticket_query / ticket_create / password_reset / escalate
    intent: str

    # 问题分析的当前主结果（用于喂给 generate_answer）
    analysis: str

    # 是否需要 KB 检索（由 analyze_question 判断；用户可在前端覆盖）
    needs_kb: bool

    # RAG 检索到的知识库上下文（needs_kb=False 时为空字符串）
    retrieved_context: str

    # 检索命中的文档来源（文件名）
    kb_sources: list[str]

    # 工单相关信息
    ticket_info: dict

    # 当前用户信息（从企业 SSO 或请求参数获取）
    user_id: str
    user_name: str
    user_department: str

    # 是否需要人工介入
    needs_human: bool

    # 回答依据来源: "kb" | "model" | ""
    answer_source: str

    # 问题分析的候选项（不同角度）—— 用于在 UI 上展示备选思路
    analysis_alternatives: list[str]

    # 生成回答的风格提示（concise / detailed / step_by_step / 空表示默认）
    style_hint: str
