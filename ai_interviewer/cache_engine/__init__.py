"""上下文缓存引擎 - 语义缓存、命中率统计"""

from ai_interviewer.cache_engine.embedding import EmbeddingService, get_embedding_service
from ai_interviewer.cache_engine.metrics import CacheMetrics, get_cache_metrics
from ai_interviewer.cache_engine.semantic_cache import SemanticCache


class CacheEngineFacade:
    """缓存引擎门面，统一对外接口（预热、统计）"""

    def __init__(self) -> None:
        self._cache = SemanticCache()
        self._metrics = get_cache_metrics()

    @property
    def cache(self) -> SemanticCache:
        return self._cache

    async def warmup(self, resume_id: str, question: str, answer: str) -> None:
        """预热：将单条问答写入缓存"""
        await self._cache.set(resume_id, question, answer)

    async def get_stats(self) -> dict:
        """获取缓存统计"""
        stats = self._metrics.get_stats()
        return stats.model_dump()


_cache_engine: CacheEngineFacade | None = None


def get_cache_engine() -> CacheEngineFacade:
    """获取缓存引擎门面单例"""
    global _cache_engine
    if _cache_engine is None:
        _cache_engine = CacheEngineFacade()
    return _cache_engine


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "SemanticCache",
    "CacheMetrics",
    "get_cache_metrics",
    "CacheEngineFacade",
    "get_cache_engine",
]
