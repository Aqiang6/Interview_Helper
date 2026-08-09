"""面试 Agent 服务 - 基于简历生成面试问题并模拟面试官对话"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ai_interviewer.cache_engine.semantic_cache import SemanticCache
from ai_interviewer.cache_engine.summarizer import SummarizationBuffer
from ai_interviewer.config import get_settings
from ai_interviewer.models import ConversationMessage
from ai_interviewer.rag_engine.retriever import RAGRetriever, get_retriever

logger = logging.getLogger(__name__)


@dataclass
class InterviewSession:
    """面试会话"""
    session_id: str
    resume_text: str
    resume_summary: str
    candidate_name: str = ""
    skills: list[str] = field(default_factory=list)
    messages: list[ConversationMessage] = field(default_factory=list)
    current_topic: str = ""
    question_count: int = 0
    max_questions: int = 15
    started_at: float = field(default_factory=time.time)
    is_finished: bool = False
    score: int = 0  # 面试评分 0-100


# 面试官系统提示词
_INTERVIEW_SYSTEM_PROMPT = """你是一位资深技术面试官，正在进行一场技术面试。

## 你的职责
1. **智能判断面试方向**：根据候选人简历内容自动判断面试重点方向
2. **深度拷打**：对核心技能进行深度追问，问到候选人答不上为止
3. **每次只问一个问题**，问题要具体、有深度，层层递进
4. 根据候选人的回答进行追问，直到获取足够深度的信息
5. 语气专业但友好，像真实的面试官一样

## 面试方向判断规则
- 如果简历包含「语义缓存」「Token管理」「RAG」「Embedding」「LLM」「Prompt Engineering」等 AI Agent 相关关键词 → **重点考察 AI Agent 工程化能力**
- 如果简历包含「分布式锁」「分库分表」「高并发」「消息队列」「微服务」等传统后端架构关键词 → **重点考察分布式系统设计能力**

## 提问策略（AI Agent 方向）
1. **语义缓存与向量匹配**：如何设计语义缓存？相似度阈值如何设定？LRU+TTL如何实现？
2. **Token管理与上下文压缩**：如何控制Token消耗？压缩策略是什么？如何保证信息保真？
3. **RAG与检索增强**：向量数据库选型？检索策略？如何处理上下文窗口限制？
4. **Prompt Engineering**：系统提示词设计原则？如何优化提示词？
5. **多模型适配与降级**：如何实现模型热切换？降级策略是什么？
6. **可观测性**：如何监控LLM调用？关键指标有哪些？

## 提问策略（传统后端方向）
1. **分布式锁**：基于Redis/Redisson实现分布式锁的原理？如何防止死锁？如何实现锁续期？
2. **高并发订单系统**：如何解决超卖问题？订单创建与座位扣减如何异步解耦？
3. **分库分表**：基于ShardingSphere如何实现水平分片？分片策略如何选择？
4. **消息队列**：如何保证消息不丢失？如何实现消息幂等？如何处理消息积压？
5. **缓存策略**：LFU与LRU的区别？缓存击穿/穿透/雪崩如何解决？热点数据如何处理？
6. **接口限流**：基于Sentinel如何实现QPS限流？如何设计自适应限流策略？

## 简历信息
{resume_summary}

## 知识库参考（RAG 检索结果，作为提问参考，不要照搬，可自由发挥追问方向）
{rag_context}

## 当前进度
已提问 {question_count}/{max_questions} 个问题
当前话题: {current_topic}

## 输出格式
直接输出你要说的话（面试官的话），不要加任何前缀或格式标记。
如果是第一个问题，先做简短的自我介绍然后开始提问。"""

# 评估系统提示词
_EVALUATE_PROMPT = """请根据以下面试对话，对候选人进行综合评估。

## 面试记录
{conversation}

## 简历信息
{resume_summary}

## 评估重点
根据简历内容自动判断评估维度：

### AI Agent 方向（简历包含语义缓存、Token管理、RAG、Embedding、LLM等关键词）
- 语义缓存与向量匹配能力
- Token管理与上下文压缩策略
- RAG与检索增强设计
- Prompt Engineering 技巧
- 多模型适配与降级方案
- 可观测性设计

### 传统后端方向（简历包含分布式锁、分库分表、高并发、消息队列、微服务等关键词）
- 分布式系统设计能力
- 高并发处理能力
- 数据库设计与优化
- 消息队列与异步处理
- 缓存策略与优化
- 系统稳定性与可靠性

请以 JSON 格式输出评估结果：
{{
    "total_score": 0-100的总分,
    "dimensions": {{
        "分布式系统设计": "0-100的评分",
        "高并发处理能力": "0-100的评分",
        "数据库设计与优化": "0-100的评分",
        "消息队列与异步处理": "0-100的评分",
        "缓存策略与优化": "0-100的评分"
    }},
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"],
    "recommendation": "通过/待定/不通过",
    "comment": "总体评价（100字以内）"
}}

只输出 JSON，不要其他内容。"""


class InterviewAgent:
    """面试 Agent - 管理面试会话和 LLM 交互"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        cache: SemanticCache | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._cache = cache
        self._summarizer = SummarizationBuffer()
        self._sessions: dict[str, InterviewSession] = {}
        self._settings = get_settings()
        self._retriever: RAGRetriever | None = None
        if self._settings.rag_enabled:
            self._retriever = get_retriever()

    def create_session(self, session_id: str, resume_text: str, resume_summary: str,
                       candidate_name: str = "", skills: list[str] | None = None) -> InterviewSession:
        """创建新的面试会话"""
        session = InterviewSession(
            session_id=session_id,
            resume_text=resume_text,
            resume_summary=resume_summary,
            candidate_name=candidate_name,
            skills=skills or [],
        )
        # 添加系统消息（首次不检索 RAG，因为话题未确定）
        session.messages.append(ConversationMessage(
            role="system",
            content=self._build_system_prompt(session),
        ))
        self._sessions[session_id] = session
        return session

    async def get_first_question(self, session_id: str) -> str:
        """获取面试的第一个问题"""
        session = self._sessions.get(session_id)
        if not session:
            return "面试会话不存在"

        # 先尝试缓存命中
        if self._cache and session.skills:
            cache_key = f"{session_id}:first_question"
            cached = await self._cache.get(session_id, "面试开场白和技术第一个问题")
            if cached.hit:
                session.messages.append(ConversationMessage(role="assistant", content=cached.answer))
                session.question_count += 1
                session.current_topic = session.skills[0] if session.skills else "技术基础"
                return cached.answer

        # 调用 LLM
        response = await self._call_llm(session)
        if response:
            session.messages.append(ConversationMessage(role="assistant", content=response))
            session.question_count += 1
            session.current_topic = session.skills[0] if session.skills else "技术基础"
            # 写入缓存
            if self._cache:
                await self._cache.set(session_id, "面试开场白和技术第一个问题", response)
            return response
        return "你好！我是今天的面试官。让我们开始吧，请先简单介绍一下你自己？"

    async def answer_and_ask(self, session_id: str, candidate_answer: str) -> str:
        """候选人回答问题，面试官追问或提下一个问题"""
        session = self._sessions.get(session_id)
        if not session:
            return "面试会话不存在"
        if session.is_finished:
            return "面试已结束"

        # 记录候选人回答
        session.messages.append(ConversationMessage(role="user", content=candidate_answer))

        # 检查是否需要压缩对话历史
        compressed = await self._summarizer.maybe_compress(
            [m for m in session.messages if m.role != "system"]
        )
        if compressed:
            logger.info("对话已压缩: %d -> %d tokens", compressed.original_tokens, compressed.compressed_tokens)

        # 检查是否达到最大问题数
        if session.question_count >= session.max_questions:
            session.is_finished = True
            return await self._generate_final_feedback(session)

        # 更新系统提示词（注入 RAG 检索的知识上下文）
        rag_context = await self._get_rag_context(session)
        session.messages[0].content = self._build_system_prompt(session, rag_context)

        # 调用 LLM 获取面试官回复
        response = await self._call_llm(session)
        if response:
            session.messages.append(ConversationMessage(role="assistant", content=response))
            session.question_count += 1
            return response
        return "好的，请继续。能详细说说你在这个方面的经验吗？"

    async def get_evaluation(self, session_id: str) -> dict:
        """生成面试评估报告"""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "面试会话不存在"}

        # 构建对话记录
        conversation_lines: list[str] = []
        for msg in session.messages:
            if msg.role == "assistant":
                conversation_lines.append(f"面试官: {msg.content}")
            elif msg.role == "user":
                conversation_lines.append(f"候选人: {msg.content}")
        conversation_text = "\n\n".join(conversation_lines)

        eval_prompt = _EVALUATE_PROMPT.format(
            conversation=conversation_text[:6000],
            resume_summary=session.resume_summary[:1000],
        )

        try:
            result = await self._call_llm_raw([
                {"role": "user", "content": eval_prompt},
            ], temperature=0.3)
            # 尝试解析 JSON
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            return json.loads(result)
        except Exception as e:
            logger.error("评估生成失败: %s", e)
            return {
                "total_score": 0,
                "comment": f"评估生成失败: {e}",
                "recommendation": "待定",
            }

    async def _call_llm(self, session: InterviewSession) -> str:
        """调用 LLM 获取面试官回复"""
        messages = [{"role": m.role, "content": m.content} for m in session.messages]
        return await self._call_llm_raw(messages, temperature=0.7)

    async def _call_llm_raw(self, messages: list[dict], temperature: float = 0.7) -> str:
        """底层 LLM 调用"""
        if not self._api_key:
            return "[未配置 API Key，请在设置页面配置]"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1500,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            return "[模型调用超时，请检查网络或 API 配置]"
        except httpx.HTTPStatusError as e:
            return f"[模型调用失败: HTTP {e.response.status_code}]"
        except Exception as e:
            return f"[模型调用失败: {e}]"

    def _build_system_prompt(self, session: InterviewSession, rag_context: str = "") -> str:
        """构建面试官系统提示词"""
        return _INTERVIEW_SYSTEM_PROMPT.format(
            resume_summary=session.resume_summary,
            rag_context=rag_context or "（无相关知识库参考）",
            question_count=session.question_count,
            max_questions=session.max_questions,
            current_topic=session.current_topic or "尚未开始",
        )

    async def _get_rag_context(self, session: InterviewSession) -> str:
        """检索知识库，返回格式化的参考知识上下文"""
        if not self._retriever:
            return ""
        try:
            result = await self._retriever.retrieve_by_skills(
                skills=session.skills,
                current_topic=session.current_topic,
            )
            return result.format_context()
        except Exception as e:
            logger.warning("RAG 检索失败，跳过: %s", e)
            return ""

    async def _generate_final_feedback(self, session: InterviewSession) -> str:
        """生成面试结束反馈"""
        session.is_finished = True
        eval_result = await self.get_evaluation(session.session_id)
        score = eval_result.get("total_score", 0)
        recommendation = eval_result.get("recommendation", "待定")
        comment = eval_result.get("comment", "")

        return (
            f"好的，今天的面试就到这里。感谢你的时间！\n\n"
            f"--- 面试评估 ---\n"
            f"综合评分: {score}/100\n"
            f"建议: {recommendation}\n"
            f"评价: {comment}"
        )

    def get_session(self, session_id: str) -> InterviewSession | None:
        """获取面试会话"""
        return self._sessions.get(session_id)

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        """获取会话消息（不含 system）"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [m for m in session.messages if m.role != "system"]
