"""Agent 工具集 - Tool Use

LangGraph Agent 可调用的工具，每个工具封装一个独立能力：
1. search_knowledge_base：检索 RAG 知识库
2. analyze_candidate_response：评估候选人回答质量（LLM 实时深度分析）

注意：已移除 select_next_topic 和 adjust_difficulty 两个工具——
- 话题切换由 LangGraph 路由节点 `next_topic` + `make_route_decision` 直接处理（二维决策表），不再需要工具形式暴露
- 难度调整直接通过 `response_quality` 体现在 `generate_question` 的 system prompt 策略里，不单独调用工具
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ai_interviewer.rag_engine.retriever import get_retriever

logger = logging.getLogger(__name__)


# ── 回答质量评估 LLM 系统提示词 ──
_QUALITY_EVAL_SYSTEM = """你是一位资深技术面试官，负责深度评估候选人的技术回答质量。

## 评估维度（每项 0-25 分，总分 100 分）
1. **技术准确性**（25分）：回答中的技术概念、原理、实现是否准确无误
2. **回答深度**（25分）：是否深入到底层原理、源码实现、设计权衡，而非停留在表面
3. **逻辑完整性**（25分）：论证是否有条理、因果清晰、覆盖全面、无明显遗漏
4. **工程实践**（25分）：是否结合实际场景、给出落地方案、提到踩坑经验或优化手段

## 输出要求
严格输出 JSON 格式，不要任何额外文字或 markdown 标记：
{{
  "total_score": 0-100,
  "level": "high|medium|low",
  "dimensions": {{
    "accuracy": 0-25,
    "depth": 0-25,
    "logic": 0-25,
    "practice": 0-25
  }},
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "has_code_example": true/false,
  "has_real_project": true/false
}}

## 评分标准
- high（75-100）：回答深入准确，有底层原理和工程实践，逻辑严谨
- medium（40-74）：回答基本正确但深度不足，或缺少实际案例，或有部分遗漏
- low（0-39）：回答错误多、过于简略、逻辑混乱或明显跑题"""


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
                    lambda: asyncio.run(retriever.retrieve(query))
                )
                result = future.result()
        else:
            result = loop.run_until_complete(retriever.retrieve(query))
        context = result.format_context()
        return context if context.strip() else "未找到相关知识"
    except Exception as e:
        logger.warning("知识库检索失败: %s", e)
        return f"知识库检索失败: {e}"


@tool
def analyze_candidate_response(
    response: str,
    topic: str,
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "",
) -> str:
    """分析候选人回答的质量，返回评估结果（LLM 实时深度分析）。

    评估维度：技术准确性、回答深度、逻辑完整性、工程实践。
    面试官根据评估结果决定是否追问或切换话题。

    Args:
        response: 候选人的回答内容
        topic: 当前面试话题
        api_key: 用户传入的模型 API Key（优先用用户配置；为空则读全局配置兜底）
        base_url: 用户传入的 Base URL
        model: 用户选择的模型名（优先用用户配置；为空则用全局 openai_model）
    """
    # 优先用用户传入的配置，其次读全局 settings 兜底
    # 注意：任何回答长度都直接进 LLM（包括"不知道"/<20字符），不再走规则 low 兜底；
    # LLM 不可用时直接报错（API Key 错误没必要强行继续）。
    from ai_interviewer.config import get_settings
    settings = get_settings()
    effective_api_key = api_key or settings.openai_api_key
    effective_base_url = base_url or settings.openai_base_url
    effective_model = model or settings.openai_model or "gpt-4o-mini"

    # 调用 LLM 进行实时深度分析
    try:
        llm = ChatOpenAI(
            api_key=effective_api_key,
            base_url=effective_base_url,
            model=effective_model,
            temperature=0.1,
            max_tokens=800,
        )
        user_prompt = f"""## 当前面试话题
{topic}

## 候选人回答
{response}

请按照评估维度严格评分并输出 JSON。"""

        result = llm.invoke([
            SystemMessage(content=_QUALITY_EVAL_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        eval_data = _parse_quality_json(result.content)
        return _format_quality_result(eval_data)

    except (RuntimeError, ValueError):
        # JSON 解析错误已包装为 ValueError / 上层已有 RuntimeError，直接透传
        raise
    except Exception as e:
        err_type = type(e).__name__
        if not effective_api_key:
            masked_key = "(空)"
        elif len(effective_api_key) <= 8:
            masked_key = "***" + (effective_api_key[-4:] if len(effective_api_key) > 4 else "")
        else:
            masked_key = effective_api_key[:8] + "..." + effective_api_key[-4:]
        msg = (
            f"候选人回答评估失败，面试无法继续：[{err_type}] {str(e) or '(无错误消息)'}。"
            f" | model={effective_model} | base_url={effective_base_url} | api_key={masked_key}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e


# 注意：已删除 _rule_based_fallback。回答评估必须走 LLM，任何失败直接抛错（不再规则兜底）。



def _parse_quality_json(content: str) -> dict:
    """解析 LLM 返回的评估 JSON，容忍 markdown 代码块和多余文本"""
    content = (content or "").strip()

    # 去除 markdown 代码块
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            content = inner

    # 直接解析
    try:
        data = json.loads(content)
        return _normalize_quality_data(data)
    except json.JSONDecodeError:
        pass

    # 兜底：截取最外层 JSON
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end + 1])
            return _normalize_quality_data(data)
        except json.JSONDecodeError:
            pass

    # JSON 解析失败直接抛错，不再伪造默认 medium 分数
    msg = (
        "候选人回答评估失败：LLM 返回内容无法解析为合法 JSON 结构。"
        f"原始内容预览：{repr(content[:200]) if content else '(空)'}"
    )
    logger.error(msg)
    raise ValueError(msg)


def _normalize_quality_data(data: dict) -> dict:
    """规范化 LLM 返回的评估数据，保证字段完整"""
    dimensions = data.get("dimensions", {}) or {}
    total = data.get("total_score")
    if total is None:
        total = sum(dimensions.values()) if dimensions else 50

    # 维度分数兜底
    for key in ["accuracy", "depth", "logic", "practice"]:
        if key not in dimensions or not isinstance(dimensions[key], (int, float)):
            dimensions[key] = total // 4

    level = data.get("level")
    if level not in ("high", "medium", "low"):
        level = "high" if total >= 75 else "medium" if total >= 40 else "low"

    return {
        "total_score": max(0, min(100, int(total))),
        "level": level,
        "dimensions": {k: max(0, min(25, int(v))) for k, v in dimensions.items()},
        "strengths": data.get("strengths", []) or [],
        "weaknesses": data.get("weaknesses", []) or [],
        "suggestions": data.get("suggestions", []) or [],
        "has_code_example": bool(data.get("has_code_example", False)),
        "has_real_project": bool(data.get("has_real_project", False)),
    }


def _format_quality_result(r: dict) -> str:
    """将结构化评估结果格式化为可读字符串（同时嵌入 JSON 便于状态读取）"""
    dim = r["dimensions"]
    parts = [
        f"质量评估[{r['level']}] 总分:{r['total_score']}/100",
        f"维度: 准确{dim['accuracy']}/25 · 深度{dim['depth']}/25 · 逻辑{dim['logic']}/25 · 实践{dim['practice']}/25",
    ]
    if r["strengths"]:
        parts.append(f"亮点: {'；'.join(r['strengths'][:2])}")
    if r["weaknesses"]:
        parts.append(f"不足: {'；'.join(r['weaknesses'][:2])}")
    flags = []
    if r["has_code_example"]:
        flags.append("含代码示例")
    if r["has_real_project"]:
        flags.append("含项目经验")
    if flags:
        parts.append("特征: " + "、".join(flags))
    # 把 JSON 附加在末尾，供 evaluate_response 节点反向解析
    parts.append("__JSON__:" + json.dumps(r, ensure_ascii=False))
    return " | ".join(parts)



# ═══════════════════════════════════════════════════════════
# 已移除 select_next_topic & adjust_difficulty 两个 @tool
# - select_next_topic 的职责由 LangGraph 路由节点 next_topic + TopicPrioritizer/make_route_decision 承担
# - adjust_difficulty 的职责由 evaluate_response 产出的 response_quality 直接驱动 system prompt 策略
# ═══════════════════════════════════════════════════════════

# 导出所有工具
ALL_TOOLS = [
    search_knowledge_base,
    analyze_candidate_response,
]

