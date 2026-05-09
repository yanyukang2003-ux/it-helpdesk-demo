"""
MCP (Model Context Protocol) Server 层
======================================

简化版 MCP 协议实现，提供：
- 工具注册与发现 (tools/list)
- 工具调用 (tools/call)
- 集成 RBAC 权限控制
- 结构化审计日志

生产环境可替换为完整的 MCP SDK 实现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from app.security import (
    UserContext,
    Role,
    check_permission,
    audit_logger,
    desensitize,
)
from app.tools import ALL_TOOLS, TOOL_MAP


# ==================================================
# Tool registry
# ==================================================

@dataclass
class MCPTool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    handler: Callable
    required_role: str = "user"

    def to_list_entry(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
            },
        }


# Tool parameter schemas (JSON Schema format)
TOOL_SCHEMAS = {
    "query_ticket": {
        "ticket_id": {
            "type": "string",
            "description": "工单编号，格式如 TK-1024",
        },
    },
    "query_my_tickets": {
        "user_id": {
            "type": "string",
            "description": "用户工号",
        },
    },
    "create_ticket": {
        "title": {"type": "string", "description": "工单标题"},
        "description": {"type": "string", "description": "问题详细描述"},
        "category": {"type": "string", "description": "分类: 硬件/软件/网络/账号/其他"},
        "priority": {"type": "string", "description": "优先级: 低/中/高/紧急"},
    },
    "reset_password": {
        "user_id": {"type": "string", "description": "用户工号"},
        "system_name": {"type": "string", "description": "系统名称: 邮箱/VPN/OA/ERP"},
    },
}


def _build_registry() -> dict[str, MCPTool]:
    registry = {}
    for t in ALL_TOOLS:
        name = t.name
        from app.security import TOOL_PERMISSIONS, Role
        required = TOOL_PERMISSIONS.get(name, Role.USER)
        registry[name] = MCPTool(
            name=name,
            description=t.description,
            parameters=TOOL_SCHEMAS.get(name, {}),
            handler=t,
            required_role=required.value,
        )
    return registry


MCP_REGISTRY = _build_registry()


# ==================================================
# MCP Protocol handlers
# ==================================================

class MCPServer:
    """简化版 MCP Server。

    处理 MCP 协议的标准请求：
    - tools/list: 返回所有可用工具
    - tools/call: 执行指定工具（含权限校验）
    """

    def __init__(self):
        self.registry = MCP_REGISTRY

    def handle_request(self, method: str, params: dict, user: UserContext, session_id: str = "") -> dict:
        """处理 MCP 请求入口。

        Args:
            method: MCP 方法名 (tools/list, tools/call)
            params: 方法参数
            user: 当前用户上下文
            session_id: 会话 ID

        Returns:
            MCP 协议响应
        """
        if method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return self._call_tool(params, user, session_id)
        else:
            return {"error": {"code": -32601, "message": f"未知方法: {method}"}}

    def _list_tools(self) -> dict:
        tools = [t.to_list_entry() for t in self.registry.values()]
        return {"tools": tools}

    def _call_tool(self, params: dict, user: UserContext, session_id: str) -> dict:
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        tool = self.registry.get(tool_name)
        if not tool:
            audit_logger.log_tool_call(user, tool_name, tool_args, "not found", False, session_id)
            return {"error": {"code": -32602, "message": f"未知工具: {tool_name}"}}

        # RBAC 权限校验
        permitted = check_permission(user, tool_name)
        if not permitted:
            audit_logger.log_tool_call(user, tool_name, tool_args, "permission denied", False, session_id)
            return {
                "error": {
                    "code": -32003,
                    "message": f"权限不足: {tool_name} 需要 {tool.required_role} 角色",
                }
            }

        # Execute tool via LangChain invoke
        try:
            result = tool.handler.invoke(tool_args)
            result_str = str(result)
            # Desensitize output
            result_str = desensitize(result_str)
            audit_logger.log_tool_call(user, tool_name, tool_args, result_str, True, session_id)
            return {
                "content": [{"type": "text", "text": result_str}],
            }
        except Exception as e:
            audit_logger.log_tool_call(user, tool_name, tool_args, str(e), False, session_id)
            return {
                "error": {"code": -32000, "message": f"工具执行失败: {str(e)}"},
                "content": [{"type": "text", "text": f"执行失败: {str(e)}"}],
            }


# Global MCP server instance
mcp_server = MCPServer()
