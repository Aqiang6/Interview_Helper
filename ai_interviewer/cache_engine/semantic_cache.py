"""语义缓存核心 - 基于向量相似度的 LLM 上下文缓存"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

import numpy as np

from ai_interviewer.cache_engine.embedding import EmbeddingService, get_embedding_service
from ai_interviewer.cache_engine.metrics import CacheMetrics, get_cache_metrics
from ai_interviewer.config import get_settings
from ai_interviewer.models import CacheEntry, CacheHitResult

logger = logging.getLogger(__name__)


class SemanticCache:
    """语义缓存引擎

    - 使用 Embedding 向量做语义相似度匹配
    - 内存存储使用 OrderedDict 实现 LRU 淘汰
    - 支持 TTL 自动过期和按 resume_id 批量失效
    - 线程安全（asyncio.Lock）
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        metrics: CacheMetrics | None = None,
    ) -> None:
        settings = get_settings()
        self._embedding = embedding_service or get_embedding_service()
        self._metrics = metrics or get_cache_metrics()
        self._threshold: float = settings.cache_similarity_threshold
        self._ttl: int = settings.cache_ttl_seconds
        self._max_size: int = settings.cache_max_size

        # 主存储：cache_key -> CacheEntry
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        # 向量索引：cache_key -> 问题向量（用于相似度计算）
        self._vectors: dict[str, np.ndarray] = {}
        # resume_id -> set[cache_key]，用于按简历批量失效
        self._resume_index: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_cache_key(resume_id: str, vector: np.ndarray) -> str:
        """生成缓存 Key = hash(resume_id + 语义向量的 hex 表示)"""
        vec_hex = vector.tobytes().hex()
        raw = f"{resume_id}:{vec_hex}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, resume_id: str, question: str) -> CacheHitResult:
        """查询缓存：计算问题语义向量，在该 resume_id 的缓存条目中查找最相似条目"""
        query_vec = await self._embedding.embed(question)

        async with self._lock:
            # 获取该 resume_id 关联的所有 cache_key
            candidate_keys = self._resume_index.get(resume_id, set())

            best_key: str | None = None
            best_similarity: float = 0.0

            now = time.time()
            expired_keys: list[str] = []

            for key in candidate_keys:
                entry = self._store.get(key)
                if entry is None:
                    continue

                # 检查 TTL 过期
                if now - entry.created_at > entry.ttl:
                    expired_keys.append(key)
                    continue

                # 计算相似度
                stored_vec = self._vectors.get(key)
                if stored_vec is None:
                    continue

                similarity = self._embedding.compute_similarity(query_vec, stored_vec)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key

            # 清理过期条目
            for key in expired_keys:
                self._remove_entry(key)
                self._metrics.record_eviction()

            # 判断是否命中
            if best_key is not None and best_similarity >= self._threshold:
                entry = self._store[best_key]
                entry.hit_count += 1
                # LRU：将命中的条目移到最后（最近使用）
                self._store.move_to_end(best_key)
                self._metrics.record_hit()
                logger.debug(
                    "缓存命中 resume_id=%s, similarity=%.4f, key=%s",
                    resume_id, best_similarity, best_key[:12],
                )
                return CacheHitResult(
                    hit=True,
                    answer=entry.answer,
                    confidence=best_similarity,
                    cache_key=best_key,
                )

            self._metrics.record_miss()
            return CacheHitResult(hit=False)

    async def set(self, resume_id: str, question: str, answer: str) -> None:
        """存入缓存"""
        query_vec = await self._embedding.embed(question)
        cache_key = self._make_cache_key(resume_id, query_vec)

        async with self._lock:
            # 如果 key 已存在，更新它
            if cache_key in self._store:
                self._store[cache_key].answer = answer
                self._store.move_to_end(cache_key)
                return

            # LRU 淘汰：超过 max_size 时淘汰最久未使用的（最前面的）
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._remove_entry(evicted_key)
                self._metrics.record_eviction()
                logger.debug("LRU 淘汰: %s", evicted_key[:12])

            # 创建新条目
            entry = CacheEntry(
                cache_key=cache_key,
                resume_id=resume_id,
                question=question,
                answer=answer,
                confidence=1.0,
                created_at=time.time(),
                ttl=self._ttl,
            )
            self._store[cache_key] = entry
            self._vectors[cache_key] = query_vec

            # 维护 resume 索引
            if resume_id not in self._resume_index:
                self._resume_index[resume_id] = set()
            self._resume_index[resume_id].add(cache_key)

            logger.debug(
                "缓存写入 resume_id=%s, key=%s, store_size=%d",
                resume_id, cache_key[:12], len(self._store),
            )

    async def invalidate_by_resume(self, resume_id: str) -> int:
        """按简历 ID 批量失效，返回被移除的条目数"""
        async with self._lock:
            keys_to_remove = self._resume_index.pop(resume_id, set())
            count = 0
            for key in keys_to_remove:
                if key in self._store:
                    self._remove_entry(key)
                    count += 1
                    self._metrics.record_eviction()
            logger.info(
                "简历 %s 缓存失效，移除 %d 条", resume_id, count,
            )
            return count

    def _remove_entry(self, key: str) -> None:
        """从所有内部数据结构中移除一个条目（调用者需持有锁）"""
        self._store.pop(key, None)
        self._vectors.pop(key, None)
        # 清理 resume 索引中的引用
        for resume_keys in self._resume_index.values():
            resume_keys.discard(key)
