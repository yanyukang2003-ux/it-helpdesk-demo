"""
知识库导入脚本
将 knowledge_base/ 目录下的文档导入向量数据库

用法:
    python scripts/ingest_docs.py
"""

import sys
import os

# 将项目根目录加入 Python 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 无论从哪个目录执行脚本，都从项目根目录加载 .env
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

if not (os.getenv("OPENAI_API_KEY") or "").strip():
    print(
        "错误: 未检测到 OPENAI_API_KEY。\n\n"
        "请在本项目根目录创建 .env 文件并设置密钥，例如：\n"
        "  cp .env.example .env\n"
        "然后编辑 .env，将 OPENAI_API_KEY= 改为你自己的 OpenAI API Key。\n"
        "也可以在当前终端执行: export OPENAI_API_KEY='sk-...'\n"
    )
    sys.exit(1)

from app.rag import build_knowledge_base


def main():
    print("📚 开始导入知识库文档...\n")
    result = build_knowledge_base()
    if result:
        print("\n🎉 知识库导入完成！可以启动 Agent 了。")
    else:
        print("\n⚠️  未找到文档，请在 knowledge_base/ 目录下添加 .md 文件。")


if __name__ == "__main__":
    main()
