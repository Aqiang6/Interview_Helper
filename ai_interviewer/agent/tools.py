"""Agent 工具集 - Tool Use

LangGraph Agent 可调用的工具，每个工具封装一个独立能力：
1. search_knowledge_base：检索 RAG 知识库
2. analyze_candidate_response：评估候选人回答质量
3. select_next_topic：选择下一个面试话题
4. adjust_difficulty：调整问题难度
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ai_interviewer.rag_engine.retriever import get_retriever

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(query: str) -> str:
    """检索技术知识库，返回与查询相关的面试题和参考答案。

    当面试官需要针对某个技术点深入提问时调用此工具。

    Args:
        query: 技术关键词，如 "Redis分布式锁" "JVM垃圾回收"
    """
    retriever = get_retriever()
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已有事件循环中，创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(retriever.retrieve_by_topic(query))
                )
                result = future.result()
        else:
            result = loop.run_until_complete(retriever.retrieve_by_topic(query))
        return result.format_context() if result.context else "未找到相关知识"
    except Exception as e:
        logger.warning("知识库检索失败: %s", e)
        return f"知识库检索失败: {e}"


@tool
def analyze_candidate_response(response: str, topic: str) -> str:
    """分析候选人回答的质量，返回评估结果。

    评估维度：技术准确性、深度、逻辑性、表达能力。
    面试官根据评估结果决定是否追问或切换话题。

    Args:
        response: 候选人的回答内容
        topic: 当前面试话题
    """
    # 基于规则的质量评估（轻量级，不调用 LLM）
    score = 0
    indicators = []

    # 技术关键词覆盖
    tech_keywords = {
        "redis": ["持久化", "RDB", "AOF", "集群", "哨兵", "缓存穿透", "雪崩"],
        "java": ["JVM", "GC", "类加载", "内存模型", "并发", "线程池", "锁"],
        "分布式": ["CAP", "一致性", "Raft", "Paxos", "分布式锁", "幂等"],
        "数据库": ["索引", "事务", "ACID", "MVCC", "锁", "优化", "分库分表"],
        "spring": ["IoC", "AOP", "Bean", "事务管理", "自动配置"],
    }

    response_lower = response.lower()
    for category, keywords in tech_keywords.items():
        if category in topic.lower() or category in response_lower:
            matched = [kw for kw in keywords if kw in response]
            score += len(matched) * 10
            if matched:
                indicators.append(f"提到关键概念: {', '.join(matched)}")

    # 回答长度（简短回答通常质量较低）
    if len(response) > 200:
        score += 20
        indicators.append("回答较详细")
    elif len(response) < 50:
        indicators.append("回答过于简短")
        score = max(0, score - 10)

    # 逻辑连接词
    logic_words = ["因为", "所以", "首先", "其次", "另外", "因此", "导致", "原理是"]
    logic_count = sum(1 for w in logic_words if w in response)
    if logic_count > 0:
        score += logic_count * 5
        indicators.append(f"逻辑清晰（{logic_count}个连接词）")

    level = "high" if score >= 60 else "medium" if score >= 30 else "low"

    return f"质量评估[{level}] 得分:{score} | {' | '.join(indicators) if indicators else '无明显技术亮点'}"


@tool
def select_next_topic(covered_topics: str, skills: str) -> str:
    """根据已覆盖的话题和候选人技能，选择下一个面试话题。

    Args:
        covered_topics: 已问答过的话题，逗号分隔
        skills: 候选人技能列表，逗号分隔
    """
    covered = set(t.strip().lower() for t in covered_topics.split(",") if t.strip())
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]

    # 话题优先级映射
    topic_priority = {
        "redis": 1, "java": 2, "spring": 3, "数据库": 4,
        "分布式": 5, "消息队列": 6, "docker": 7, "kafka": 8,
        "python": 9, "ai agent": 10, "rag": 11,
    }

    # 从候选人技能中选择未覆盖的最高优先级话题
    candidates = []
    for skill in skill_list:
        skill_lower = skill.lower()
        if skill_lower not in covered:
            priority = topic_priority.get(skill_lower, 99)
            candidates.append((priority, skill))

    if not candidates:
        return "所有技能话题已覆盖完毕，可以进入评估阶段"

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@tool
def adjust_difficulty(candidate_level: str, current_difficulty: str) -> str:
    """根据候选人水平调整问题难度。

    Args:
        candidate_level: 候选人评估水平 "junior" / "mid" / "senior"
        current_difficulty: 当前难度 "easy" / "medium" / "hard"
    """
    levels = {"junior": 1, "mid": 2, "senior": 3}
    difficulties = {"easy": 1, "medium": 2, "hard": 3}

    level_num = levels.get(candidate_level, 2)
    diff_num = difficulties.get(current_difficulty, 2)

    # 根据候选人水平调整
    if level_num > diff_num:
        new_diff = min(3, diff_num + 1)
    elif level_num < diff_num:
        new_diff = max(1, diff_num - 1)
    else:
        new_diff = diff_num

    diff_map = {1: "easy", 2: "medium", 3: "hard"}
    return diff_map[new_diff]


# 导出所有工具
ALL_TOOLS = [
    search_knowledge_base,
    analyze_candidate_response,
    select_next_topic,
    adjust_difficulty,
]
