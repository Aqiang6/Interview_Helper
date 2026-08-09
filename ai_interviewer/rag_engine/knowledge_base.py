"""知识库管理 - 加载技术文档、分块、构建向量索引"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_interviewer.cache_engine.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

# 知识库数据目录
_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class KnowledgeItem:
    """单条知识条目"""
    id: str
    category: str        # 分类: java / distributed / database / ai_agent / ...
    topic: str           # 主题: 如 "Redis分布式锁"
    question: str        # 面试题
    answer: str          # 参考答案
    keywords: list[str] = field(default_factory=list)  # 关键词
    _vector: np.ndarray | None = None  # 延迟赋值的向量

    @property
    def vector(self) -> np.ndarray | None:
        return self._vector

    @vector.setter
    def vector(self, value: np.ndarray) -> None:
        self._vector = value

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "topic": self.topic,
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "keywords": self.keywords,
        }


class KnowledgeBase:
    """技术知识库 - 管理知识条目和向量索引"""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embedding = embedding_service or get_embedding_service()
        self._items: list[KnowledgeItem] = []
        self._vectors: np.ndarray | None = None  # [N, D] 矩阵
        self._index: dict[str, KnowledgeItem] = {}  # id -> item
        self._loaded = False

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._vectors is not None

    async def load(self) -> None:
        """从 data 目录加载所有 JSON 知识文件并构建向量索引"""
        if self._loaded:
            return

        # 加载所有 JSON 文件
        items: list[KnowledgeItem] = []
        if _DATA_DIR.exists():
            for json_file in sorted(_DATA_DIR.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    for entry in data:
                        items.append(KnowledgeItem(
                            id=f"{entry['category']}_{entry['id']}",
                            category=entry["category"],
                            topic=entry["topic"],
                            question=entry["question"],
                            answer=entry["answer"],
                            keywords=entry.get("keywords", []),
                        ))
                    logger.info("加载知识文件: %s (%d 条)", json_file.name, len(data))
                except Exception as e:
                    logger.error("加载知识文件失败 %s: %s", json_file.name, e)

        self._items = items
        self._index = {item.id: item for item in items}
        logger.info("知识库加载完成，共 %d 条", len(items))

        # 构建向量索引
        if items:
            texts = [self._build_search_text(item) for item in items]
            vectors = await self._embedding.embed_batch(texts)
            self._vectors = np.stack(vectors)  # [N, D]
            for item, vec in zip(items, vectors):
                item.vector = vec
            logger.info("知识库向量索引构建完成: %s", self._vectors.shape)

        self._loaded = True

    @staticmethod
    def _build_search_text(item: KnowledgeItem) -> str:
        """构建用于向量化的搜索文本"""
        parts = [item.topic, item.question]
        if item.keywords:
            parts.append(" ".join(item.keywords))
        return " | ".join(parts)

    def get_by_category(self, category: str) -> list[KnowledgeItem]:
        """按分类获取知识条目"""
        return [item for item in self._items if item.category == category]

    def get_categories(self) -> list[str]:
        """获取所有分类"""
        return sorted(set(item.category for item in self._items))

    def to_dict(self) -> dict:
        """知识库概览"""
        cats: dict[str, int] = {}
        for item in self._items:
            cats[item.category] = cats.get(item.category, 0) + 1
        return {
            "total": len(self._items),
            "categories": cats,
            "is_ready": self.is_ready,
        }


# ── 全局单例 ──

_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """获取 KnowledgeBase 全局单例"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
