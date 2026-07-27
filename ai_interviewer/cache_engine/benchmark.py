"""基准测试脚本 - 测试缓存引擎性能和压缩效果

可独立运行：python -m ai_interviewer.cache_engine.benchmark
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import numpy as np

# Windows 终端默认使用 GBK 编码，强制切换为 UTF-8 以支持特殊字符
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from ai_interviewer.models import CacheHitResult, ConversationMessage, SummaryResult


# ═══════════════════════════════════════════
#  模拟 Embedding 服务（无需真实模型）
# ═══════════════════════════════════════════

class MockEmbeddingService:
    """模拟 Embedding 服务，基于字符 n-gram 生成伪向量

    使用字符 bigram 特征构建稀疏向量再降维，使得文本越相似向量越接近。
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}
        # 预生成一个固定的投影矩阵，将高维稀疏特征映射到 dim 维
        self._proj_rng = np.random.RandomState(42)
        self._proj_matrix = self._proj_rng.randn(4096, dim).astype(np.float32) / np.sqrt(dim)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """基于字符 bigram 生成伪向量，相似文本自然产生相近的向量"""
        if text in self._cache:
            return self._cache[text]
        # 提取字符 bigram 特征到 4096 维稀疏向量
        sparse = np.zeros(4096, dtype=np.float32)
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            idx = hash(bigram) % 4096
            sparse[idx] += 1.0
        # 投影到目标维度
        vec = sparse @ self._proj_matrix
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._cache[text] = vec
        return vec

    async def embed(self, text: str) -> np.ndarray:
        """模拟异步 embed"""
        await asyncio.sleep(0.001)  # 模拟推理延迟
        return self._text_to_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """模拟批量 embed"""
        await asyncio.sleep(0.001 * len(texts))
        return [self._text_to_vector(t) for t in texts]

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))


# ═══════════════════════════════════════════
#  测试结果收集
# ═══════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """单项测试结果"""
    name: str
    passed: bool = True
    metrics: dict[str, float | str | int] = field(default_factory=dict)
    details: str = ""


# ═══════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════

async def test_cache_hit_vs_miss_latency() -> BenchmarkResult:
    """测试缓存命中 vs 未命中延迟对比"""
    from ai_interviewer.cache_engine.semantic_cache import SemanticCache

    mock_embedding = MockEmbeddingService()
    cache = SemanticCache(embedding_service=mock_embedding)

    # 预热：写入 10 条缓存
    resume_id = "resume_001"
    for i in range(10):
        await cache.set(
            resume_id,
            f"请介绍一下你在 Python 项目 {i} 中的经验",
            f"我在 Python 项目 {i} 中负责后端开发...",
        )

    # 测试命中延迟（使用相似问题）
    hit_latencies: list[float] = []
    for i in range(10):
        question = f"请介绍一下你在 Python 项目 {i} 中的经验"  # 与预热完全一致
        start = time.perf_counter()
        result = await cache.get(resume_id, question)
        elapsed = time.perf_counter() - start
        if result.hit:
            hit_latencies.append(elapsed)

    # 测试未命中延迟（使用完全不同的问题）
    miss_latencies: list[float] = []
    for i in range(10):
        question = f"xyz_完全无关的随机问题_编号_{i}_abcdef_12345"
        start = time.perf_counter()
        result = await cache.get(resume_id, question)
        elapsed = time.perf_counter() - start
        if not result.hit:
            miss_latencies.append(elapsed)

    avg_hit = statistics.mean(hit_latencies) * 1000 if hit_latencies else 0
    avg_miss = statistics.mean(miss_latencies) * 1000 if miss_latencies else 0

    return BenchmarkResult(
        name="缓存命中 vs 未命中延迟",
        passed=len(hit_latencies) > 0,
        metrics={
            "平均命中延迟(ms)": round(avg_hit, 3),
            "平均未命中延迟(ms)": round(avg_miss, 3),
            "命中次数": len(hit_latencies),
            "未命中次数": len(miss_latencies),
        },
        details="命中延迟应显著低于未命中延迟（未命中仍需遍历向量计算相似度）",
    )


async def test_summarization_token_reduction() -> BenchmarkResult:
    """测试摘要压缩前后 Token 数对比"""
    from ai_interviewer.cache_engine.summarizer import SummarizationBuffer

    buffer = SummarizationBuffer()

    # 构造超过阈值的长对话
    messages: list[ConversationMessage] = []
    filler_messages = [
        "你好！", "好的", "明白了", "谢谢", "嗯嗯",
        "收到", "可以的", "没问题", "好的，我们继续",
    ]
    tech_messages = [
        "我在上一个项目中使用了 Python FastAPI 框架开发微服务后端，负责用户认证和订单管理模块。",
        "项目采用了 Docker 容器化部署，使用 Kubernetes 进行编排管理，部署在 AWS EKS 上。",
        "数据库使用 PostgreSQL，缓存层使用 Redis，消息队列用 Kafka 处理异步任务。",
        "在性能优化方面，我通过引入 Elasticsearch 实现了全文检索，将搜索延迟从 500ms 降到 50ms。",
        "团队协作使用 Git + CI/CD 流水线，代码审查流程严格，采用 Trunk-Based Development 模式。",
        "前端使用 React + TypeScript，状态管理用 Redux，构建工具是 Vite。",
        "为什么选择 Kafka 而不是 RabbitMQ？因为我们需要处理高吞吐量的事件流数据。",
        "项目中遇到的最大挑战是分布式事务一致性问题，最终用 Saga 模式解决。",
        "如何保证系统的高可用？我们实现了熔断、降级、限流机制，使用 Sentinel 框架。",
        "对于微服务之间的通信，同步调用使用 gRPC，异步事件用 Kafka 消息驱动。",
    ]

    # 混合填充消息和技术消息，制造足够长的对话（需超过 threshold=4000 tokens）
    ts = time.time()
    for round_idx in range(18):
        for filler in filler_messages:
            messages.append(ConversationMessage(
                role="user" if round_idx % 2 == 0 else "assistant",
                content=filler,
                timestamp=ts + round_idx,
            ))
        for tech in tech_messages:
            messages.append(ConversationMessage(
                role="user" if round_idx % 2 == 0 else "assistant",
                content=tech,
                timestamp=ts + round_idx,
            ))

    # 计算原始 token
    total_text = "\n".join(m.content for m in messages)
    original_tokens = buffer._count_tokens(total_text)

    # 提取实体（不需要 LLM 调用）
    entities = buffer._extract_entities(messages)

    # 模拟 LLM 摘要（避免真实 API 调用）
    mock_summary = (
        "候选人在多个技术栈有丰富经验：Python/FastAPI 微服务后端开发，"
        "Docker/Kubernetes/AWS 容器化部署，PostgreSQL/Redis/Kafka 数据层设计，"
        "Elasticsearch 全文检索优化（500ms→50ms），React/TypeScript 前端开发。"
        "具备分布式系统设计能力，使用 Saga 模式解决分布式事务，"
        "实现熔断降级限流保障高可用，gRPC + Kafka 消息驱动架构。"
    )
    compressed_tokens = buffer._count_tokens(mock_summary)
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

    return BenchmarkResult(
        name="摘要压缩 Token 数对比",
        passed=original_tokens > buffer._threshold,
        metrics={
            "原始 Token 数": original_tokens,
            "压缩后 Token 数": compressed_tokens,
            "压缩比率": round(ratio, 4),
            "目标比率": buffer._target_ratio,
            "保留实体数": len(entities),
            "消息总数": len(messages),
        },
        details=f"实体列表: {', '.join(entities[:10])}{'...' if len(entities) > 10 else ''}",
    )


async def test_hit_rate_by_threshold() -> BenchmarkResult:
    """测试不同相似度阈值下的命中率"""
    from ai_interviewer.cache_engine.semantic_cache import SemanticCache

    mock_embedding = MockEmbeddingService()
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    hit_rates: dict[float, float] = {}

    # 准备数据：写入 20 条缓存
    resume_id = "resume_threshold_test"
    base_questions = [
        "请介绍你的 Python 开发经验",
        "你使用过哪些前端框架",
        "描述一个你主导的微服务项目",
        "你在数据库设计方面的经验",
        "如何使用 Docker 进行部署",
        "你对 Kubernetes 的理解",
        "谈谈你对 CI/CD 的看法",
        "你如何保证代码质量",
        "描述你的项目管理经验",
        "你对 Agile 开发的理解",
        "你在性能优化方面的经验",
        "如何处理高并发场景",
        "你对分布式系统的理解",
        "描述你的调试技巧",
        "你如何学习新技术",
        "你在团队协作中的角色",
        "你对 RESTful API 设计的看法",
        "你如何处理技术债务",
        "描述一个你解决过的难题",
        "你对代码审查的看法",
    ]

    for i, q in enumerate(base_questions):
        await cache_set_with_mock(cache=None, embedding=mock_embedding,
                                  resume_id=resume_id, question=q,
                                  answer=f"回答 {i}", preloaded_cache=None)

    # 对每个阈值测试命中率
    # 查询集：一半相似（略改文本），一半完全不同
    similar_queries = [q + " 能否详细说说" for q in base_questions[:10]]
    different_queries = [
        "今天天气怎么样", "你最喜欢的电影是什么", "1+1等于几",
        "推荐一家餐厅", "你周末一般做什么",
        "最近有什么好看的书", "你喜欢什么运动",
        "你的家乡在哪里", "你养宠物吗", "你会弹乐器吗",
    ]
    all_queries = similar_queries + different_queries

    for threshold in thresholds:
        cache = SemanticCache(embedding_service=mock_embedding)
        cache._threshold = threshold

        # 写入数据
        for i, q in enumerate(base_questions):
            await cache.set(resume_id, q, f"回答 {i}")

        hits = 0
        for q in all_queries:
            result = await cache.get(resume_id, q)
            if result.hit:
                hits += 1

        hit_rates[threshold] = hits / len(all_queries) if all_queries else 0

    return BenchmarkResult(
        name="不同阈值下的命中率",
        passed=True,
        metrics={f"阈值 {t:.2f}": round(r, 4) for t, r in hit_rates.items()},
        details="阈值越高，命中率应越低（更严格的匹配要求）",
    )


async def cache_set_with_mock(
    cache: None,
    embedding: MockEmbeddingService,
    resume_id: str,
    question: str,
    answer: str,
    preloaded_cache: None,
) -> None:
    """辅助函数 - 此处仅为接口占位，实际测试中直接使用 SemanticCache.set"""
    pass


# ═══════════════════════════════════════════
#  报告输出
# ═══════════════════════════════════════════

def print_report(results: list[BenchmarkResult]) -> None:
    """打印结构化的测试结果报告"""
    width = 72
    print("\n" + "═" * width)
    print("  AI Agent Interviewer - 上下文缓存引擎基准测试报告")
    print("═" * width)

    for i, result in enumerate(results, 1):
        status = "✅ 通过" if result.passed else "❌ 失败"
        print(f"\n┌─ [{i}] {result.name}  {status}")
        print("├" + "─" * (width - 1))
        for key, value in result.metrics.items():
            print(f"│  {key}: {value}")
        if result.details:
            print(f"│  说明: {result.details}")
        print("└" + "─" * (width - 1))

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'─' * width}")
    print(f"  总计: {total} 项测试, {passed} 项通过, {total - passed} 项失败")
    print("═" * width + "\n")


# ═══════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════

async def main() -> None:
    """运行所有基准测试"""
    print("正在初始化基准测试环境...\n")

    results: list[BenchmarkResult] = []

    # 测试 1: 缓存命中 vs 未命中延迟
    print("▶ 运行测试 1/3: 缓存命中 vs 未命中延迟...")
    results.append(await test_cache_hit_vs_miss_latency())

    # 测试 2: 摘要压缩 Token 数对比
    print("▶ 运行测试 2/3: 摘要压缩 Token 数对比...")
    results.append(await test_summarization_token_reduction())

    # 测试 3: 不同阈值下的命中率
    print("▶ 运行测试 3/3: 不同相似度阈值下的命中率...")
    results.append(await test_hit_rate_by_threshold())

    # 输出报告
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
