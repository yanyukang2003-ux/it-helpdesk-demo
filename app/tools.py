"""
工具定义模块
定义 Agent 可以调用的所有外部工具

注意：当前使用模拟数据，生产环境中需要替换为真实的 API 调用
"""

from datetime import datetime
from langchain_core.tools import tool


# ==================================================
# 模拟数据（生产环境替换为真实数据库/API）
# ==================================================

MOCK_TICKETS = {
    "TK-1024": {
        "id": "TK-1024",
        "title": "MacBook 屏幕闪烁",
        "status": "处理中",
        "assignee": "张工",
        "created": "2026-03-28",
        "updated": "2026-03-30",
        "notes": "已预约周三上门维修",
    },
    "TK-1019": {
        "id": "TK-1019",
        "title": "邮箱无法发送附件",
        "status": "已解决",
        "assignee": "李工",
        "created": "2026-03-25",
        "updated": "2026-03-26",
        "notes": "附件大小超过限制，已调整为 50MB",
    },
    "TK-1015": {
        "id": "TK-1015",
        "title": "VPN 连接频繁断开",
        "status": "待处理",
        "assignee": "未分配",
        "created": "2026-03-29",
        "updated": "2026-03-29",
        "notes": "已排入队列，预计明天处理",
    },
}


# ==================================================
# 工单查询工具
# ==================================================

@tool
def query_ticket(ticket_id: str) -> str:
    """根据工单 ID 查询工单状态。当用户询问某个具体工单的进度时使用。
    参数 ticket_id: 工单编号，格式如 TK-1024"""
    ticket_id = ticket_id.strip().upper()
    ticket = MOCK_TICKETS.get(ticket_id)

    if not ticket:
        return f"未找到工单 {ticket_id}，请确认工单号是否正确。"

    return (
        f"工单号: {ticket['id']}\n"
        f"标题: {ticket['title']}\n"
        f"状态: {ticket['status']}\n"
        f"处理人: {ticket['assignee']}\n"
        f"创建时间: {ticket['created']}\n"
        f"最近更新: {ticket['updated']}\n"
        f"备注: {ticket['notes']}"
    )


@tool
def query_my_tickets(user_id: str) -> str:
    """查询某个用户的所有工单。当用户问"我的工单"但没提供具体工单号时使用。
    参数 user_id: 用户工号"""
    # 模拟返回该用户的所有工单
    # 生产环境：调用工单系统 API，按 user_id 过滤
    return (
        "你当前有以下工单:\n"
        "1. TK-1024 - MacBook 屏幕闪烁 - 状态: 处理中\n"
        "2. TK-1019 - 邮箱无法发送附件 - 状态: 已解决\n"
        "3. TK-1015 - VPN 连接频繁断开 - 状态: 待处理"
    )


# ==================================================
# 工单创建工具
# ==================================================

@tool
def create_ticket(
    title: str,
    description: str,
    category: str,
    priority: str = "中",
) -> str:
    """创建新的 IT 支持工单。当用户报告新问题需要 IT 支持时使用。
    参数:
      title: 工单标题，简要描述问题
      description: 问题的详细描述
      category: 分类，可选值: 硬件/软件/网络/账号/其他
      priority: 优先级，可选值: 低/中/高/紧急
    """
    # 生产环境：调用工单系统 API 创建工单
    new_id = f"TK-{datetime.now().strftime('%m%d%H%M')}"
    return (
        f"✅ 工单创建成功！\n"
        f"工单号: {new_id}\n"
        f"标题: {title}\n"
        f"描述: {description}\n"
        f"分类: {category}\n"
        f"优先级: {priority}\n"
        f"IT 团队将在 2 小时内响应。"
    )


# ==================================================
# 密码重置工具
# ==================================================

@tool
def reset_password(user_id: str, system_name: str) -> str:
    """为用户重置指定系统的密码。
    参数:
      user_id: 用户工号
      system_name: 系统名称，如 邮箱/VPN/OA/ERP"""
    # 生产环境：调用各系统的密码重置 API
    # 注意：这是一个敏感操作，生产环境中应该加入二次确认机制
    return (
        f"✅ 已为用户 {user_id} 重置 {system_name} 密码。\n"
        f"临时密码已发送至用户手机，有效期 24 小时，请尽快登录修改。"
    )


# ==================================================
# 工具注册表（方便统一管理）
# ==================================================

ALL_TOOLS = [query_ticket, query_my_tickets, create_ticket, reset_password]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
