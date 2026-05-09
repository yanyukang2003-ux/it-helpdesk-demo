"""
配置管理模块
从 .env 文件或环境变量中读取配置
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置"""

    # LLM 模型
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gpt-4o")

    # 向量数据库
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")

    # LangGraph checkpoint：留空则用内存（进程重启丢失）；设为 sqlite 路径可持久化并支持跨重启分支
    LANGGRAPH_CHECKPOINT_SQLITE: str = os.getenv("LANGGRAPH_CHECKPOINT_SQLITE", "").strip()

    # RAG 参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVER_K: int = 5  # 返回 top-k 个文档块（扩大给 Re-rank）
    RETRIEVER_FETCH_K: int = 10  # MMR 检索时先取 top-k 再筛选

    # Agent 参数
    CONFIDENCE_THRESHOLD: float = 0.4  # RAG 置信度低于此阈值转人工
    EVALUATION_THRESHOLD: float = 0.7  # PGE 评估综合得分低于此阈值触发重生成
    MAX_ITERATIONS: int = 10  # Agent 最大循环次数
    MAX_REGENERATIONS: int = 3  # PGE 反思循环最大重生成次数

    # 滚动分层压缩记忆
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "./memory.db")
    MEMORY_WORKING_TURNS: int = 4       # Working memory 保留最近 N 轮完整对话
    MEMORY_COMPRESSION_INTERVAL: int = 4  # 每 N 轮触发一次压缩
    MEMORY_MAX_FACTS: int = 20          # 长期事实库容量上限
    MEMORY_MAX_COMPRESSED: int = 10     # 压缩摘要条数上限

    # 服务
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))


config = Config()
