"""缓存命中率统计"""

from __future__ import annotations

import threading

from ai_interviewer.models import CacheStats


class CacheMetrics:
    """缓存命中率统计（线程安全单例）"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def record_hit(self) -> None:
        """记录一次缓存命中"""
        with self._lock:
            self._hits += 1

    def record_miss(self) -> None:
        """记录一次缓存未命中"""
        with self._lock:
            self._misses += 1

    def record_eviction(self) -> None:
        """记录一次缓存淘汰"""
        with self._lock:
            self._evictions += 1

    def get_stats(self) -> CacheStats:
        """获取当前统计数据"""
        with self._lock:
            total = self._hits + self._misses
            return CacheStats(
                total_queries=total,
                cache_hits=self._hits,
                cache_misses=self._misses,
                evictions=self._evictions,
            )

    def reset(self) -> None:
        """重置所有统计计数"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


# ── 全局单例 ──

_cache_metrics: CacheMetrics | None = None
_metrics_lock = threading.Lock()


def get_cache_metrics() -> CacheMetrics:
    """获取 CacheMetrics 全局单例"""
    global _cache_metrics
    if _cache_metrics is None:
        with _metrics_lock:
            if _cache_metrics is None:
                _cache_metrics = CacheMetrics()
    return _cache_metrics
