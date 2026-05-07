"""
Agent 测试用例
运行: pytest tests/test_agent.py -v
"""

import pytest
from langchain_core.messages import HumanMessage
from app.graph import build_graph


@pytest.fixture(scope="module")
def agent():
    """创建不带记忆的 Agent 实例（测试用）"""
    return build_graph(with_memory=False)


def _invoke_agent(agent, user_input: str, user_id: str = "EMP001") -> dict:
    """辅助函数：调用 Agent 并返回结果"""
    return agent.invoke({
        "messages": [HumanMessage(content=user_input)],
        "user_id": user_id,
        "user_name": "测试用户",
        "user_department": "工程部",
        "needs_human": False,
        "confidence": 1.0,
        "intent": "",
        "retrieved_context": "",
        "ticket_info": {},
    })


# ==================================================
# 意图分类测试
# ==================================================

class TestIntentClassification:
    """测试意图分类是否准确"""

    def test_knowledge_qa_intent(self, agent):
        result = _invoke_agent(agent, "VPN 怎么连接？")
        assert result["intent"] == "knowledge_qa"

    def test_ticket_query_intent(self, agent):
        result = _invoke_agent(agent, "帮我查一下工单 TK-1024 的状态")
        assert result["intent"] == "ticket_query"

    def test_ticket_create_intent(self, agent):
        result = _invoke_agent(agent, "我的电脑开不了机了，需要维修")
        assert result["intent"] == "ticket_create"

    def test_password_reset_intent(self, agent):
        result = _invoke_agent(agent, "我忘记邮箱密码了")
        assert result["intent"] == "password_reset"

    def test_escalate_intent(self, agent):
        result = _invoke_agent(agent, "我要找真人客服")
        assert result["intent"] == "escalate"


# ==================================================
# 回复质量测试
# ==================================================

class TestResponseQuality:
    """测试回复内容是否包含关键信息"""

    def test_ticket_query_contains_id(self, agent):
        result = _invoke_agent(agent, "查一下 TK-1024")
        reply = result["messages"][-1].content
        assert "TK-1024" in reply

    def test_ticket_create_has_confirmation(self, agent):
        result = _invoke_agent(agent, "打印机坏了，打不出来东西")
        reply = result["messages"][-1].content
        assert "TK-" in reply  # 应包含新工单号

    def test_escalate_has_guidance(self, agent):
        result = _invoke_agent(agent, "转人工")
        reply = result["messages"][-1].content
        assert "人工" in reply or "客服" in reply

    def test_password_reset_has_result(self, agent):
        result = _invoke_agent(agent, "帮我重置 VPN 密码")
        reply = result["messages"][-1].content
        assert "密码" in reply


# ==================================================
# 边界情况测试
# ==================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_like_input(self, agent):
        """模糊输入不应导致崩溃"""
        result = _invoke_agent(agent, "嗯")
        assert result["messages"][-1].content  # 有回复就行

    def test_long_input(self, agent):
        """长输入不应导致崩溃"""
        long_text = "我的电脑有问题 " * 100
        result = _invoke_agent(agent, long_text)
        assert result["messages"][-1].content

    def test_english_input(self, agent):
        """英文输入也应能处理"""
        result = _invoke_agent(agent, "How do I connect to VPN?")
        assert result["messages"][-1].content
