"""
RAG 知识库模块
负责文档导入、向量化存储、检索、置信度评估
"""

import json
import re

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config
from app.prompts import RAG_CONFIDENCE_PROMPT, RERANK_PROMPT, QUERY_REWRITE_PROMPT

confidence_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)
rerank_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)
rewrite_llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.0)


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
            "fetch_k": config.RETRIEVER_FETCH_K * 2,  # 扩大初筛池给 Re-rank
        },
    )


def rewrite_query(query: str) -> list[str]:
    """将口语化查询改写为 3 个更适合检索的变体。

    - 变体1: 核心关键词提取
    - 变体2: 操作步骤角度
    - 变体3: 概念定义角度
    """
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    try:
        resp = rewrite_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
        variants = data.get("variants") or []
        # 去重 + 保留原文
        seen = {query}
        result = [query]
        for v in variants:
            v = str(v).strip()[:50]
            if v and v not in seen:
                result.append(v)
                seen.add(v)
        return result[:3]
    except Exception:
        return [query]  # 降级：原样返回


def _retrieve_docs(retriever, query: str) -> list:
    """单次检索，封装异常处理。"""
    try:
        return retriever.invoke(query)
    except Exception:
        return []


def _merge_deduplicate(doc_lists: list[list], top_n: int = 8) -> list:
    """合并多次检索结果，按内容去重，保持多样性。"""
    seen_contents = set()
    merged = []
    for docs in doc_lists:
        for d in docs:
            key = d.page_content[:100]
            if key not in seen_contents:
                seen_contents.add(key)
                merged.append(d)
                if len(merged) >= top_n:
                    return merged
    return merged


def re_rank(query: str, docs: list, top_k: int = 3) -> list:
    """Cross-Encoder 风格的 LLM 精排。

    对 MMR 初筛结果用 LLM 重新打分排序，
    区分「操作指南」和「故障排查」，优先返回能回答问题的文档。
    """
    if len(docs) <= 1:
        return docs

    # 构建候选列表
    candidates_text = ""
    for i, doc in enumerate(docs):
        preview = doc.page_content[:300].replace("\n", " ")
        candidates_text += f"[{i}] {preview}\n\n"

    prompt = RERANK_PROMPT.format(query=query, candidates=candidates_text[:3000])

    try:
        resp = rerank_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
    except Exception as e:
        print(f"⚠️ Re-rank 失败: {e}")
        return docs[:top_k]  # 降级：返回原始排序的前 k 条

    scores = data.get("scores") or []
    if not scores:
        return docs[:top_k]

    # 按分数降序排列
    score_map = {}
    for item in scores:
        idx = int(item.get("id", -1))
        score = float(item.get("score", 5))
        score_map[idx] = score

    if not score_map:
        return docs[:top_k]

    ranked = sorted(enumerate(docs), key=lambda x: score_map.get(x[0], 0), reverse=True)

    # 日志
    if ranked:
        old_first = docs[0].page_content[:50].replace("\n", " ")
        new_first = ranked[0][1].page_content[:50].replace("\n", " ")
        if old_first != new_first:
            print(f"🔄 Re-rank 调序: {old_first[:40]}... → {new_first[:40]}...")

    return [doc for _, doc in ranked[:top_k]]


def retrieve_with_sources(query: str, use_rerank: bool = True) -> tuple[str, list[str]]:
    """
    全链路检索：Query 改写 → 多路召回 → 合并去重 → GraphRAG 跨文档 → Re-rank 精排。

    Args:
        query: 用户原始查询
        use_rerank: 是否启用 LLM Re-rank 精排
    """
    import os as _os
    import re as _re

    retriever = get_retriever()

    # Phase 1: Query Rewriting — 1 个问题 → 3 个改写变体
    variants = rewrite_query(query)
    print(f"🔍 Query 改写: {query[:40]} → {[v[:30] for v in variants[1:]]}")

    # Phase 2: Multi-variant retrieval — 每个变体分别检索
    all_doc_lists = []
    for v in variants:
        docs = _retrieve_docs(retriever, v)
        if docs:
            all_doc_lists.append(docs)

    if not all_doc_lists:
        return "未找到相关知识库文档。", []

    # Phase 3: Merge + Deduplicate
    docs = _merge_deduplicate(all_doc_lists)

    # Phase 4: GraphRAG — 跨文档多跳检索
    matched_sources = set()
    for d in docs:
        src = d.metadata.get("source", "")
        if src:
            matched_sources.add(_os.path.basename(src))

    if len(matched_sources) < 2:
        # 尝试跨文档扩展
        from app.graphrag import get_related_docs
        # 从已召回的文档内容中提取关键词作为匹配实体
        content_text = " ".join(d.page_content[:200] for d in docs[:2])
        # 简单实体匹配：提取大写缩写、中文专有名词
        entities = _re.findall(r'[A-Z][a-zA-Z-]{2,}|[A-Z]{3,}', content_text)
        entities += _re.findall(r'[一-鿿]{2,4}(?:系统|网络|邮箱|密码|客户端|验证|服务|文档)', content_text)
        related = get_related_docs(query, list(set(entities)))
        if related:
            # 对相关文档做补充检索
            extra_docs = []
            for doc_name in related:
                extra = _retrieve_docs(retriever, doc_name.replace(".md", "").replace("_", " "))
                extra_docs.extend(extra)
            if extra_docs:
                docs = _merge_deduplicate([docs] + [extra_docs])

    # Phase 5: Re-rank
    if use_rerank and len(docs) > 1:
        docs = re_rank(query, docs)

    context_parts: list[str] = []
    sources: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "未知来源")
        short = _os.path.basename(src) if src else "未知来源"
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


# --------------------------------------------------
# 置信度独立计算
# --------------------------------------------------

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


def calculate_confidence(query: str, context: str, sources: list[str]) -> dict:
    """独立计算 RAG 检索置信度。

    基于检索内容的相关性、完整性、来源可靠性三个维度评估，
    低置信度时自动标记应转人工处理。
    """
    if not context or context.startswith("未找到") or context.startswith("知识库检索失败"):
        return {
            "relevance": 0.0,
            "completeness": 0.0,
            "reliability": 0.0,
            "confidence": 0.0,
            "should_escalate": True,
            "reason": "知识库未命中相关文档，建议转人工处理",
        }

    prompt = RAG_CONFIDENCE_PROMPT.format(
        question=query,
        context=context[:3000],
        sources=", ".join(sources) if sources else "无",
    )

    try:
        resp = confidence_llm.invoke(prompt)
        data = _extract_json(str(resp.content)) or {}
    except Exception as e:
        print(f"⚠️ 置信度评估失败: {e}")
        heuristic_conf = min(0.6, len(sources) * 0.2)
        return {
            "relevance": heuristic_conf,
            "completeness": heuristic_conf,
            "reliability": heuristic_conf,
            "confidence": heuristic_conf,
            "should_escalate": heuristic_conf < config.CONFIDENCE_THRESHOLD,
            "reason": f"评估失败，启发式估算 (命中 {len(sources)} 个来源)",
        }

    relevance = max(0.0, min(1.0, float(data.get("relevance", 0.5))))
    completeness = max(0.0, min(1.0, float(data.get("completeness", 0.5))))
    reliability = max(0.0, min(1.0, float(data.get("reliability", 0.5))))
    confidence = round(relevance * 0.5 + completeness * 0.3 + reliability * 0.2, 4)

    should_escalate = confidence < config.CONFIDENCE_THRESHOLD or bool(data.get("should_escalate", False))
    reason = str(data.get("reason", ""))[:200]

    return {
        "relevance": relevance,
        "completeness": completeness,
        "reliability": reliability,
        "confidence": confidence,
        "should_escalate": should_escalate,
        "reason": reason,
    }
