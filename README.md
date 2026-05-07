# 🤖 IT Helpdesk Agent

基于 LangGraph 的企业内部 IT 问答机器人。

## 功能

- **知识库问答** — 基于 RAG 回答 IT 相关问题（VPN、邮箱、WiFi 等）
- **工单查询** — 查询已有工单的状态和进度
- **工单创建** — 自动创建新的 IT 支持工单
- **密码重置** — 为用户重置各系统密码
- **智能转人工** — 无法处理时自动转接人工客服

## 架构

```
用户消息 → 意图分类 → ┬→ 知识库问答 (RAG) → 置信度检查 → 回复/转人工
                       ├→ 工单查询 → 回复
                       ├→ 工单创建 → 回复
                       ├→ 密码重置 → 回复
                       └→ 转人工 → 回复
```

## 快速开始

### 1. 环境要求

- Python 3.11（推荐）
- OpenAI API Key

### 2. 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd it-helpdesk-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 必填: OPENAI_API_KEY
# 可选: LANGCHAIN_API_KEY（用于 LangSmith 追踪）
```

### 4. 导入知识库

```bash
python scripts/ingest_docs.py
```

你可以在 `knowledge_base/` 目录下添加更多 `.md` 文件，然后重新运行此脚本。

### 5. 运行

**命令行测试模式（推荐先用这个验证）：**

```bash
python main.py
```

**启动 API 服务：**

```bash
python server.py
# 或
uvicorn server:app --reload --port 8000
```

**Docker 方式：**

```bash
docker compose up --build
```

### 6. 测试

```bash
# API 测试
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "VPN 怎么连接？", "user_id": "EMP001"}'

# 单元测试
pytest tests/ -v
```

## 项目结构

```
it-helpdesk-agent/
├── app/
│   ├── config.py       # 配置管理
│   ├── state.py        # Agent 状态定义
│   ├── graph.py        # LangGraph 图定义（核心）
│   ├── nodes.py        # 各节点处理逻辑
│   ├── tools.py        # 工具定义（工单、密码重置等）
│   ├── rag.py          # RAG 检索模块
│   └── prompts.py      # Prompt 模板
├── knowledge_base/     # IT 知识库文档（Markdown）
├── scripts/
│   └── ingest_docs.py  # 知识库导入脚本
├── tests/
│   └── test_agent.py   # 测试用例
├── server.py           # FastAPI 服务
├── main.py             # 命令行交互入口
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 自定义扩展

### 添加新的知识库文档

在 `knowledge_base/` 下新建 `.md` 文件，然后运行 `python scripts/ingest_docs.py`。

### 接入真实工单系统

修改 `app/tools.py` 中的工具函数，将模拟数据替换为真实 API 调用。

### 添加新的意图和处理节点

1. 在 `app/prompts.py` 的意图分类 Prompt 中新增意图类型
2. 在 `app/nodes.py` 中实现新的处理函数
3. 在 `app/graph.py` 中注册节点和路由边

### 接入企业 IM

- **飞书**: 使用飞书开放平台的机器人 API
- **钉钉**: 使用钉钉自定义机器人 Webhook
- **企业微信**: 使用企业微信应用 API
- **Slack**: 使用 Slack Bot API

中间层只需将 IM 消息转发到 `/chat` 端点，再将回复发回即可。
