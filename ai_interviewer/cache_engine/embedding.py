"""Embedding 服务 - 基于 sentence-transformers 的文本向量化"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

from ai_interviewer.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """文本 Embedding 服务，使用 sentence-transformers 本地模型"""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_name: str = get_settings().embedding_model
        self._lock = asyncio.Lock()

    def _load_model(self) -> SentenceTransformer:
        """同步加载模型（懒加载）"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("正在加载 Embedding 模型: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding 模型加载完成")
        return self._model

    async def _ensure_model(self) -> SentenceTransformer:
        """确保模型已加载，首次调用时在线程池中执行加载"""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load_model)
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        """将单条文本转为向量"""
        model = await self._ensure_model()
        vector: np.ndarray = await asyncio.to_thread(model.encode, text, normalize_embeddings=True)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量将文本转为向量列表"""
        if not texts:
            return []
        model = await self._ensure_model()
        vectors: np.ndarray = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v for v in vectors]

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个向量的余弦相似度（向量已归一化，直接点积即可）"""
        similarity: float = float(np.dot(vec_a, vec_b))
        return similarity


# ── 全局单例 ──

_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """获取 EmbeddingService 全局单例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
