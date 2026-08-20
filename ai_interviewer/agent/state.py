"""Agent 状态定义 - LangGraph 的 TypedDict State

管理面试全流程的状态，包括：
- Memory：短期记忆（对话历史）+ 长期记忆（候选人画像）
- Planning：面试计划（话题列表 + 当前阶段）
- Context：RAG 检索结果、简历摘要
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class CandidateProfile(TypedDict, total=False):
    """候选人画像 - 长期记忆（Long-term Memory）

    在面试过程中持续更新，记录候选人的能力评估。
    """
    name: str
    skills: list[str]
    skill_scores: dict[str, int]          # 每个技能的得分 0-100
    strengths: list[str]                  # 强项
    weaknesses: list[str]                 # 弱项
    answered_topics: list[str]            # 已问答过的话题
    difficulty_level: str                 # "junior" | "mid" | "senior"
    overall_impression: str               # 整体印象


class InterviewPlan(TypedDict, total=False):
    """面试计划 - Planning

    面试开始时生成，根据简历技能 + 目标岗位 JD 动态规划面试话题和顺序。
    """
    topics: list[str]                                       # 面试话题列表（按优先级排序，标准名）
    topic_aliases: dict[str, str]                           # 标准名 → 简历原始名
    current_topic_index: int                                # 当前话题索引
    phase: str                                              # "intro" | "basics" | "deep_dive" | "wrap_up" | "evaluation"
    questions_per_topic: int                                # 遗留字段（老调用方兜底）；实际用 topics_per_skill
    # ↓ JD 驱动新增字段
    skill_importance: dict[str, str]                        # {标准话题: "required" | "preferred" | "bonus"}
    topics_per_skill: dict[str, int]                        # {"required":4, "preferred":3, "bonus":2}
    topic_question_counter: dict[str, int]                  # {标准话题: 已经问了几轮}；next_topic 时重置为 0
    position_id: str                                        # 目标岗位 ID（ai_fullstack / agent_engineer / ...）
    position_name: str                                      # 目标岗位中文名
    priority_source: str                                    # "llm" / "fallback"（规划来源，排查用）


class InterviewState(TypedDict):
    """LangGraph 面试 Agent 的完整状态

    贯穿整个面试流程，所有节点共享读写。
    """
    # ── Memory：短期记忆 ──
    messages: list[BaseMessage]           # 完整对话历史

    # ── Memory：长期记忆 ──
    candidate_profile: CandidateProfile    # 候选人画像

    # ── Planning ──
    interview_plan: InterviewPlan          # 面试计划

    # ── 简历信息 ──
    resume_summary: str                    # 简历摘要
    candidate_name: str
    skills: list[str]                       # 候选人技能列表

    # ── 目标岗位（JD 驱动新增） ──
    target_position: str                    # 前端选的 position_id 或岗位名
    jd_position_name: str                   # 岗位中文名称
    jd_raw_text: str                        # JD 完整原文（给 LLM 排序/出题时参考）

    # ── 面试进度 ──
    question_count: int                    # 已提问数
    max_questions: int                      # 最大问题数（根据岗位密度动态调整）
    current_topic: str                     # 当前面试话题
    is_finished: bool                      # 面试是否结束

    # ── RAG 上下文 ──
    rag_context: str                       # RAG 检索到的参考知识

    # ── 工具调用结果 ──
    last_tool_result: str                   # 最近一次工具调用的结果
    response_quality: str                  # 候选人最近一次回答的质量评估

    # ── LLM 配置 ──
    api_key: str
    base_url: str
    model: str
