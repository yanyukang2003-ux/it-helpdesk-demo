"""
RAG 知识库模块
负责文档导入、向量化存储、和检索
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 模型实例"""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_knowledge_base(docs_dir: str | None = None) -> Chroma | None:
    """
    将知识库文档导入向量数据库

    Args:
        docs_dir: 文档目录路径，默认使用配置中的路径

    Returns:
        Chroma 向量数据库实例
    """
    docs_dir = docs_dir or config.KNOWLEDGE_BASE_DIR

    # 加载所有 Markdown 文档
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        print(f"警告: 在 {docs_dir} 中未找到任何 .md 文件")
        return None

    # 分块策略：按标题层级分割，保留语义完整性
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ""],
    )
    chunks = splitter.split_documents(documents)

    # 存入向量数据库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    print(f"✅ 知识库构建完成：导入 {len(documents)} 个文档，生成 {len(chunks)} 个文档块")
    return vectorstore


def get_retriever():
    """
    获取检索器

    使用 MMR (Maximal Marginal Relevance) 检索策略，
    在相关性和多样性之间取平衡
    """
    vectorstore = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVER_K,
            "fetch_k": config.RETRIEVER_FETCH_K,
        },
    )


def retrieve_with_sources(query: str) -> tuple[str, list[str]]:
    """
    执行检索，同时返回上下文文本和命中文档来源列表。
    """
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"检索出错: {e}")
        return "知识库检索失败，请稍后重试。", []

    if not docs:
        return "未找到相关知识库文档。", []

    import os
    context_parts: list[str] = []
    sources: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "未知来源")
        short = os.path.basename(src) if src else "未知来源"
        if short not in sources:
            sources.append(short)
        context_parts.append(f"[文档 {i}] 来源: {src}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts), sources


def retrieve(query: str) -> str:
    """
    执行检索，返回拼接后的上下文文本

    Args:
        query: 用户的查询问题

    Returns:
        格式化的上下文字符串，包含来源信息
    """
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"检索出错: {e}")
        return "知识库检索失败，请稍后重试。"

    if not docs:
        return "未找到相关知识库文档。"

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        context_parts.append(
            f"[文档 {i}] 来源: {source}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
