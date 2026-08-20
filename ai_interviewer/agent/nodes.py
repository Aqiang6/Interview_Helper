"""Agent 节点 - LangGraph Graph Nodes

每个节点是面试流程中的一个步骤，通过 LangGraph 的条件边连接：
1. plan_interview：分析简历，规划面试话题（Planning）
2. generate_question：生成面试问题（Tool Use + RAG）
3. evaluate_response：评估候选人回答（Memory 更新，LLM 深度分析）
4. decide_next：决定下一步动作（追问/换题/结束）
5. generate_evaluation：生成最终评估报告
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from ai_interviewer.agent.state import InterviewState
from ai_interviewer.agent.topic_prioritizer import (
    get_topic_prioritizer,
    make_route_decision,
    prioritize_by_jd,
)
from ai_interviewer.agent.tools import (
    ALL_TOOLS,
    _QUALITY_EVAL_SYSTEM,
    _parse_quality_json,
)

logger = logging.getLogger(__name__)

# ── 系统提示词 ──

_INTERVIEWER_SYSTEM = """你是一位资深技术面试官，正在针对岗位「{position_name}」进行技术面试。

## 面试策略
1. 根据目标岗位 JD 决定问题方向和深度：JD 要求为【必备 required】的技能要深挖原理、源码、线上问题；JD 要求为【加分 bonus】的技能点到为止即可
2. 当前话题要求强度: {topic_importance_label}（required=必须问到深层、preferred=问到中等深度、bonus=快速扫一遍）
3. 每次只问一个问题，层层递进，深度追问
4. 根据候选人回答质量调整难度
5. 可调用工具辅助决策

## 可用工具
- search_knowledge_base：检索技术知识库，获取参考面试题和答案
- analyze_candidate_response：评估候选人回答质量

## 简历信息
{resume_summary}

## RAG 知识参考
{rag_context}

## 面试进度
阶段: {phase} | 已提问: {question_count}/{max_questions} | 当前话题: {current_topic} | 本话题已问: {topic_round}/{quota}轮
本话题 JD 要求强度: {topic_importance_label}

输出格式：直接输出面试官的话，不要加前缀。"""


_EVALUATION_SYSTEM = """你是面试评估专家。根据面试对话记录，识别候选人在哪些知识点和技术栈上存在不足、需要补强。

输出 JSON 格式：
{{
    "improvement_suggestions": ["需要补强的知识点/技术栈1", "需要补强的知识点/技术栈2"]
}}

只输出 JSON，不要任何额外文字或评分。"""


# ── 节点函数 ──

def plan_interview(state: InterviewState) -> dict:
    """节点1：规划面试流程（Planning，JD 驱动）

    主路径：调用 LLM 根据「目标岗位 JD + 简历技能 + 简历摘要」动态生成：
      - ordered_topics（按 JD 要求强度从高到低排序）
      - skill_importance（required / preferred / bonus）
      - suggested_max_questions（按岗位技术密度调整 13-20）
      - topics_per_skill（每档技能该问几轮）
    失败自动降级为旧版 TopicPrioritizer 规则排序。
    如果已有计划，则跳过（幂等）。
    """
    # 已有计划 → 跳过
    if state.get("interview_plan", {}).get("topics"):
        return {}

    skills = state.get("skills", [])
    resume_text = state.get("resume_summary", "")
    api_key = state.get("api_key", "")
    base_url = state.get("base_url", "https://api.openai.com/v1")
    model = state.get("model", "") or "gpt-4o-mini"
    jd_position_name = state.get("jd_position_name", "") or "通用技术岗"
    jd_raw_text = state.get("jd_raw_text", "")
    target_position = state.get("target_position", "")

    # ── JD 驱动 LLM 动态排序主路径（失败自动降级） ──
    jd_result = prioritize_by_jd(
        api_key=api_key,
        base_url=base_url,
        model=model,
        skills=skills,
        resume_summary=resume_text,
        jd_position_name=jd_position_name,
        jd_raw_text=jd_raw_text,
    )
    sorted_topics = jd_result.ordered_topics if jd_result.ordered_topics else ["技术基础"]
    topic_aliases = {canonical: original for original, canonical in jd_result.ordered_aliases}
    if not topic_aliases and sorted_topics:
        topic_aliases = {t: t for t in sorted_topics}
    topic_question_counter = {t: 0 for t in sorted_topics}
    max_questions = jd_result.suggested_max_questions or 15

    # 计算 phase：如果 required 技能很多 → 直接 deep_dive；否则 basics
    req_count = sum(1 for v in jd_result.skill_importance.values() if v == "required")
    phase = "deep_dive" if req_count >= 2 else "basics"

    plan = {
        "topics": sorted_topics,
        "topic_aliases": topic_aliases,
        "current_topic_index": 0,
        "phase": phase,
        "questions_per_topic": 3,  # 遗留字段兜底
        # JD 新增字段
        "skill_importance": dict(jd_result.skill_importance),
        "topics_per_skill": dict(jd_result.topics_per_skill),
        "topic_question_counter": topic_question_counter,
        "position_id": target_position,
        "position_name": jd_position_name,
        "priority_source": jd_result.source,
    }

    logger.info(
        "面试计划生成(source=%s): position=%s max_q=%d required=%d topics=%s",
        jd_result.source, jd_position_name, max_questions, req_count, sorted_topics[:10],
    )

    return {
        "interview_plan": plan,
        "current_topic": plan["topics"][0] if plan["topics"] else "技术基础",
        "max_questions": max_questions,  # 覆盖默认 15
    }


def generate_question(state: InterviewState) -> dict:
    """节点2：生成面试问题（Tool Use + RAG）

    调用 LLM 生成面试问题，系统提示词中注入 RAG 检索结果。
    """
    plan = state.get("interview_plan", {})
    messages = list(state.get("messages", []))
    question_count = state.get("question_count", 0)
    current_topic = state.get("current_topic", "")

    # RAG 检索：根据当前话题检索知识库
    rag_context = state.get("rag_context", "")
    if not rag_context and current_topic:
        try:
            from ai_interviewer.rag_engine.retriever import get_retriever
            import asyncio
            retriever = get_retriever()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(retriever.retrieve(current_topic))
                        )
                        rag_result = future.result()
                else:
                    rag_result = loop.run_until_complete(retriever.retrieve(current_topic))
                rag_context = rag_result.format_context() if rag_result.items else ""
            except RuntimeError:
                rag_context = ""
        except Exception as e:
            logger.warning("RAG 检索失败: %s", e)
            rag_context = ""

    # 计算当前话题要求强度 + 已经问了几轮（供 system prompt 显示，也供路由决策用）
    importance_map: dict[str, str] = plan.get("skill_importance", {}) or {}
    importance_label = importance_map.get(current_topic)
    if importance_label not in {"required", "preferred", "bonus"}:
        # 模糊匹配
        importance_label = "preferred"
        for k, v in importance_map.items():
            if k and (k in current_topic or current_topic in k) and v in {"required", "preferred", "bonus"}:
                importance_label = v
                break
    topics_per_skill_cfg: dict[str, int] = plan.get("topics_per_skill", {}) or {"required": 4, "preferred": 3, "bonus": 2}
    quota = int(topics_per_skill_cfg.get(importance_label, 3))

    # topic_question_counter：当前话题问了几轮（本题出完要 +1）
    counter: dict[str, int] = plan.get("topic_question_counter", {}) or {}
    topic_round_prev = int(counter.get(current_topic, 0))  # 出这道题之前的轮数

    # 构建系统提示词
    position_name = plan.get("position_name") or state.get("jd_position_name") or "通用技术岗"
    system_content = _INTERVIEWER_SYSTEM.format(
        position_name=position_name,
        resume_summary=state.get("resume_summary", "无"),
        rag_context=rag_context or "无",
        phase=plan.get("phase", "intro"),
        question_count=question_count,
        max_questions=state.get("max_questions", 15),
        current_topic=current_topic or "未开始",
        topic_importance_label=importance_label,
        topic_round=topic_round_prev,
        quota=quota,
    )

    # 替换或添加系统消息
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_content)
    else:
        messages.insert(0, SystemMessage(content=system_content))

    # 调用 LLM（temperature=0.1 减少 token/延迟，问题不需要 creative；失败直接降级提示用户检查配置）
    llm = _get_llm(state, temperature=0.1)
    try:
        response = llm.invoke(messages)
        ai_content = response.content
    except Exception as e:
        logger.error("LLM 调用失败: %s", e)
        ai_content = (
            f"（系统提示：LLM 调用失败，请检查 API Key / Base URL / 模型名称是否正确。"
            f"错误：{type(e).__name__}: {e}）"
        )

    # 更新状态
    messages.append(AIMessage(content=ai_content))
    new_count = question_count + 1
    new_phase = plan.get("phase", "intro")
    if new_count == 1:
        new_phase = "basics"
    elif new_count >= 3:
        new_phase = "deep_dive"

    # 累加当前话题已问轮数
    new_counter = dict(counter)
    if current_topic:
        new_counter[current_topic] = topic_round_prev + 1

    return {
        "messages": messages,
        "question_count": new_count,
        "rag_context": rag_context,
        "interview_plan": {
            **plan,
            "phase": new_phase,
            "topic_question_counter": new_counter,
        },
    }


def evaluate_response(state: InterviewState) -> dict:
    """节点3：评估候选人回答（LLM 实时深度分析 + Memory 更新）

    通过 LLM 从四个维度评估回答深度：技术准确性、回答深度、逻辑完整性、工程实践。
    任何失败（LLM 不可用 / JSON 解析失败）直接抛出带上下文的 RuntimeError，
    不再做规则兜底或默认 medium 分数（API Key 错了没必要强行继续）。
    """
    messages = state.get("messages", [])
    profile = dict(state.get("candidate_profile", {}))

    # 获取最近的候选人回答
    recent_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            recent_human = msg.content
            break

    if not recent_human:
        return {}

    current_topic = state.get("current_topic", "unknown")

    # 更新已回答话题
    answered = list(profile.get("answered_topics", []))
    if current_topic not in answered:
        answered.append(current_topic)
    profile["answered_topics"] = answered

    # ── LLM 深度评估 ──
    eval_data = _llm_evaluate_answer(recent_human, current_topic, state)

    # 更新技能得分（使用四维度总分映射到 0-100）
    total_score = eval_data["total_score"]
    skill_scores = dict(profile.get("skill_scores", {}))
    # 同名话题多次回答取历史最高分 + 本次加权（避免一次高分冲掉多次平庸）
    prev = skill_scores.get(current_topic, 0)
    skill_scores[current_topic] = max(prev, int(prev * 0.4 + total_score * 0.6)) if prev else total_score
    profile["skill_scores"] = skill_scores

    # 更新强弱项：按 level 判断，同时记录评估里的具体优/缺点
    strengths = list(profile.get("strengths", []))
    weaknesses = list(profile.get("weaknesses", []))
    if eval_data["level"] == "high" and current_topic not in strengths:
        strengths.append(current_topic)
    elif eval_data["level"] == "low" and current_topic not in weaknesses:
        weaknesses.append(current_topic)
    # 把 LLM 指出的具体缺点作为待补强知识点挂到 profile 上（非结构化字段存 notes）
    if eval_data["weaknesses"]:
        notes = profile.get("improvement_notes", [])
        for w in eval_data["weaknesses"] + eval_data["suggestions"]:
            note = f"[{current_topic}] {w}"
            if note not in notes:
                notes.append(note)
        profile["improvement_notes"] = notes[-30:]  # 最多保留 30 条
    profile["strengths"] = strengths
    profile["weaknesses"] = weaknesses

    return {
        "candidate_profile": profile,
        "response_quality": eval_data["level"],
    }


def _llm_evaluate_answer(answer: str, topic: str, state: InterviewState) -> dict:
    """调用 LLM 做四维度回答质量评估，任何失败直接抛 RuntimeError"""
    # 注意：任何回答长度都直接进 LLM（包括"不知道"/<20字符），不再走规则 low 兜底；
    # LLM 不可用时直接报错（API Key 错误没必要强行继续）。

    # 模型/API 配置：优先用户 state 里传入的，其次全局 settings，最后兜底
    from ai_interviewer.config import get_settings
    settings = get_settings()
    user_model = state.get("model") or settings.openai_model or "gpt-4o-mini"
    try:
        llm = ChatOpenAI(
            api_key=state.get("api_key") or settings.openai_api_key,
            base_url=state.get("base_url") or settings.openai_base_url,
            model=user_model,
            temperature=0.1,
            max_tokens=900,
        )
        prompt = f"""## 当前面试话题
{topic}

## 候选人回答
{answer}

按评估维度严格评分，输出 JSON。"""
        result = llm.invoke([
            SystemMessage(content=_QUALITY_EVAL_SYSTEM),
            HumanMessage(content=prompt),
        ])
        eval_data = _parse_quality_json(result.content)
        logger.info(
            "LLM 回答评估: topic=%s score=%d level=%s dim=%s",
            topic, eval_data["total_score"], eval_data["level"], eval_data["dimensions"],
        )
        return eval_data
    except (RuntimeError, ValueError):
        # JSON 解析错误已包装为 ValueError / 上层已有 RuntimeError，直接透传
        raise
    except Exception as e:
        err_type = type(e).__name__
        effective_api_key = state.get("api_key") or settings.openai_api_key
        effective_base_url = state.get("base_url") or settings.openai_base_url
        if not effective_api_key:
            masked_key = "(空)"
        elif len(effective_api_key) <= 8:
            masked_key = "***" + (effective_api_key[-4:] if len(effective_api_key) > 4 else "")
        else:
            masked_key = effective_api_key[:8] + "..." + effective_api_key[-4:]
        msg = (
            f"候选人回答评估失败，面试无法继续：[{err_type}] {str(e) or '(无错误消息)'}。"
            f" | topic={topic} | model={user_model} | base_url={effective_base_url}"
            f" | api_key={masked_key}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e


def decide_next(state: InterviewState) -> str:
    """节点4：条件路由 - 决定下一步动作（JD 二维决策表）

    不再是简单三句 if-else，而是「JD 要求强度 × 回答质量」二维矩阵决策：
    - required（必备技能）答崩 → 必须追问；高分才切
    - preferred（偏好技能）答崩 → 追问几次；中等以上直接切
    - bonus（加分技能）答崩也无所谓 → 直接切
    同时叠加：题数到顶结束、配额满了强制切、剩题不够覆盖时压缩追问。

    返回值与原兼容："followup" / "next_topic" / "end"
    """
    question_count = state.get("question_count", 0)
    max_questions = state.get("max_questions", 15)
    plan = state.get("interview_plan", {})
    topics = plan.get("topics", [])
    current_idx = plan.get("current_topic_index", 0)
    current_topic = state.get("current_topic", "")

    # 当前话题要求强度（required / preferred / bonus）
    importance_map: dict[str, str] = plan.get("skill_importance", {}) or {}
    importance = importance_map.get(current_topic, "preferred")
    if importance not in {"required", "preferred", "bonus"}:
        for k, v in importance_map.items():
            if k and (k in current_topic or current_topic in k) and v in {"required", "preferred", "bonus"}:
                importance = v
                break
        else:
            importance = "preferred"

    # 当前话题问了几轮（用于配额保护）
    counter: dict[str, int] = plan.get("topic_question_counter", {}) or {}
    questions_on_this_topic = int(counter.get(current_topic, 0))

    quality = state.get("response_quality", "medium")
    # followup 标记：上一轮路由决定要追问，这轮 quality 还没更新时兜底
    if quality == "followup":
        quality = "low"

    return make_route_decision(
        quality=quality,
        importance=importance,
        questions_on_this_topic=questions_on_this_topic,
        questions_per_skill=plan.get("topics_per_skill", {"required": 4, "preferred": 3, "bonus": 2}),
        question_count=question_count,
        max_questions=max_questions,
        current_topic_index=current_idx,
        total_topics=len(topics),
    )


def next_topic(state: InterviewState) -> dict:
    """节点5a：切换到下一个面试话题（同时清 RAG 上下文，避免混用上一个话题的知识库）"""
    plan = dict(state.get("interview_plan", {}))
    topics = plan.get("topics", [])
    current_idx = plan.get("current_topic_index", 0)

    new_idx = min(current_idx + 1, len(topics) - 1) if topics else 0
    new_topic = topics[new_idx] if topics else "技术基础"

    return {
        "interview_plan": {**plan, "current_topic_index": new_idx},
        "current_topic": new_topic,
        "rag_context": "",  # 切题 = 清空 RAG 上下文缓存，生成问题时重新检索新话题的资料
    }


def followup(state: InterviewState) -> dict:
    """节点5b：追问准备（标记需要追问，实际提问由 generate_question 完成）"""
    return {"response_quality": "followup"}


def generate_evaluation(state: InterviewState) -> dict:
    """节点6：生成最终评估报告"""
    messages = state.get("messages", [])
    profile = state.get("candidate_profile", {})

    # 构建对话记录
    conversation = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            conversation.append(f"面试官: {msg.content}")
        elif isinstance(msg, HumanMessage):
            conversation.append(f"候选人: {msg.content}")

    eval_prompt = f"""请根据以下面试记录，识别候选人需要补强的知识点和技术栈。

## 面试记录
{chr(10).join(conversation[:30])}

## 候选人画像
技能得分: {profile.get('skill_scores', {})}
强项: {profile.get('strengths', [])}
弱项: {profile.get('weaknesses', [])}
"""

    llm = _get_llm(state, temperature=0.3)
    try:
        result = llm.invoke([
            SystemMessage(content=_EVALUATION_SYSTEM),
            HumanMessage(content=eval_prompt),
        ])
        evaluation = _parse_evaluation_json(result.content)
    except Exception as e:
        logger.error("评估 LLM 调用失败: %s", e)
        evaluation = {
            "improvement_suggestions": [],
            "comment": f"评估生成失败（LLM 调用异常）：{type(e).__name__}: {e}",
        }

    return {
        "is_finished": True,
        "last_tool_result": json.dumps(evaluation, ensure_ascii=False),
    }


def _parse_evaluation_json(content: str) -> dict:
    """从 LLM 输出中解析评估 JSON，容忍 markdown 代码块和前后多余文本"""
    content = (content or "").strip()

    # 去除 markdown 代码块标记
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            # 去除 "json" 等语言标识
            if inner.startswith("json"):
                inner = inner[4:].strip()
            elif inner.startswith("{"):
                pass
            content = inner

    # 直接尝试解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 兜底：从内容中截取第一个完整的 JSON 对象
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = content[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return {
        "improvement_suggestions": [],
        "comment": f"评估解析失败: {content[:200]}",
    }


# ── 辅助函数 ──

def _get_llm(state: InterviewState, temperature: float = 0.7) -> ChatOpenAI:
    """创建 LangChain ChatOpenAI 实例"""
    return ChatOpenAI(
        api_key=state.get("api_key", ""),
        base_url=state.get("base_url", "https://api.openai.com/v1"),
        model=state.get("model", "gpt-4o"),
        temperature=temperature,
        max_tokens=1500,
    )
