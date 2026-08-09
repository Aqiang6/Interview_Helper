"""RAG 检索器 - 基于向量相似度检索相关知识，为面试官提供提问参考"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ai_interviewer.cache_engine.embedding import EmbeddingService, get_embedding_service
from ai_interviewer.rag_engine.knowledge_base import KnowledgeBase, KnowledgeItem, get_knowledge_base

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    items: list[KnowledgeItem]
    scores: list[float]

    def format_context(self) -> str:
        """格式化为可注入 LLM 提示词的上下文文本"""
        if not self.items:
            return ""
        parts: list[str] = []
        for i, (item, score) in enumerate(zip(self.items, self.scores), 1):
            parts.append(
                f"【参考知识 {i}】（相关度: {score:.2f}）\n"
                f"主题: {item.topic}\n"
                f"面试题: {item.question}\n"
                f"参考答案: {item.answer}"
            )
        return "\n\n".join(parts)


class RAGRetriever:
    """RAG 检索器 - 根据查询检索最相关的技术知识"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        embedding_service: EmbeddingService | None = None,
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> None:
        self._kb = knowledge_base or get_knowledge_base()
        self._embedding = embedding_service or get_embedding_service()
        self._top_k = top_k
        self._min_score = min_score

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """检索与查询最相关的知识条目

        Args:
            query: 查询文本（通常是当前面试话题 + 候选人技能）
            top_k: 返回的条目数，默认使用初始化时的值
        """
        k = top_k or self._top_k

        # 确保知识库已加载
        if not self._kb.is_ready:
            await self._kb.load()

        if not self._kb.is_ready or self._kb.size == 0:
            logger.warning("知识库未就绪或为空，跳过检索")
            return RetrievalResult(query=query, items=[], scores=[])

        # 向量检索
        query_vec = await self._embedding.embed(query)
        scores = np.dot(self._kb._vectors, query_vec)  # [N]

        # Top-K 召回
        top_indices = np.argsort(scores)[::-1][:k]

        items: list[KnowledgeItem] = []
        item_scores: list[float] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self._min_score:
                continue
            items.append(self._kb._items[idx])
            item_scores.append(score)

        logger.info("RAG 检索: query='%s', 召回 %d 条 (最高分: %.3f)",
                     query[:50], len(items), item_scores[0] if item_scores else 0)

        return RetrievalResult(query=query, items=items, scores=item_scores)

    async def retrieve_by_skills(self, skills: list[str], current_topic: str = "") -> RetrievalResult:
        """根据候选人技能和当前话题检索知识

        Args:
            skills: 候选人技能列表
            current_topic: 当前面试话题
        """
        query_parts = [current_topic] if current_topic else []
        query_parts.extend(skills[:5])  # 最多取前5个技能
        query = " ".join(query_parts)
        return await self.retrieve(query)


# ── 全局单例 ──

_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """获取 RAGRetriever 全局单例"""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
