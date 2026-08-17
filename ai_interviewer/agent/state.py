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

    面试开始时生成，根据简历技能动态规划面试话题和顺序。
    """
    topics: list[str]                     # 面试话题列表（按优先级排序）
    current_topic_index: int              # 当前话题索引
    phase: str                            # "intro" | "basics" | "deep_dive" | "wrap_up" | "evaluation"
    questions_per_topic: int              # 每个话题的最大问题数


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

    # ── 面试进度 ──
    question_count: int                    # 已提问数
    max_questions: int                      # 最大问题数
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
