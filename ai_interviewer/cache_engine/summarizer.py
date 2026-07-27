"""摘要压缩引擎 - 对长对话进行智能压缩以控制上下文窗口大小"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import tiktoken

from ai_interviewer.config import get_settings
from ai_interviewer.models import ConversationMessage, SummaryResult

logger = logging.getLogger(__name__)

# 技术栈关键词集合，用于实体提取和消息重要性判断
_TECH_KEYWORDS: set[str] = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "react", "vue", "angular", "django", "flask", "fastapi", "spring",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "kafka", "rabbitmq", "grpc", "rest", "graphql",
    "机器学习", "深度学习", "nlp", "cv", "推荐系统",
    "微服务", "分布式", "高并发", "负载均衡", "消息队列",
    "git", "ci/cd", "devops", "agile", "scrum",
    "linux", "nginx", "tomcat", "jenkins",
    "pytorch", "tensorflow", "transformer", "bert", "gpt",
}

# 项目/实体名称的匹配模式（中文项目名、英文驼峰命名等）
_PROJECT_PATTERN = re.compile(
    r"(?:项目|系统|平台|工程|服务)\s*[「《]?[\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9\-_]{1,20}[」》]?"
    r"|[\u4e00-\u9fa5]{2,8}(?:项目|系统|平台)"
    r"|[A-Z][a-zA-Z0-9]+(?:Project|System|Platform|Service|Engine)",
    re.IGNORECASE,
)

# 寒暄/冗余消息匹配模式
_FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(你好|您好|hello|hi|hey|嗯|好的|ok|okay|谢谢|感谢|辛苦了|不客气|再见|拜拜)[\s!！。.？?]*$", re.IGNORECASE),
    re.compile(r"^(明白了|了解了|收到|知道了|没问题|可以的|对|是的|没错|嗯嗯)[\s!！。.？?]*$"),
    re.compile(r"^(好的，|好，|嗯，)?(我们继续|继续吧|开始吧|下一[个题])"),
    re.compile(r"^(请问)?(你|您)(好|好呀)[\s!！。.？?]*$"),
]

# 追问链路检测：包含追问特征词
_FOLLOWUP_PATTERN = re.compile(
    r"(为什么|怎么|如何|能否|可以|详细|具体|举例|解释|深入|追问|进一步|原因|原理|底层)"
)


class SummarizationBuffer:
    """对话摘要压缩缓冲区

    - 当对话 Token 数超过阈值时触发压缩
    - 保留技术栈关键词、项目名、追问链路
    - 丢弃寒暄和重复确认
    - 使用 LLM 做最终摘要
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._threshold: int = settings.summary_token_threshold
        self._target_ratio: float = settings.summary_target_ratio
        self._api_key: str = settings.openai_api_key
        self._base_url: str = settings.openai_base_url
        self._model: str = settings.openai_model
        try:
            self._encoding = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """使用 tiktoken 计算文本的 Token 数"""
        return len(self._encoding.encode(text))

    def _extract_entities(self, messages: list[ConversationMessage]) -> list[str]:
        """基于正则和关键词提取技术栈、项目名等实体"""
        entities: set[str] = set()
        combined_text = " ".join(m.content for m in messages)

        # 提取技术栈关键词
        lower_text = combined_text.lower()
        for keyword in _TECH_KEYWORDS:
            if keyword.lower() in lower_text:
                entities.add(keyword)

        # 提取项目名称
        for match in _PROJECT_PATTERN.finditer(combined_text):
            entities.add(match.group().strip())

        return sorted(entities)

    def _is_filler(self, message: ConversationMessage) -> bool:
        """判断是否为寒暄/冗余消息"""
        content = message.content.strip()
        # 空消息视为冗余
        if not content:
            return True
        # 过短且匹配寒暄模式
        if len(content) < 30:
            for pattern in _FILLER_PATTERNS:
                if pattern.match(content):
                    return True
        return False

    def _is_important(self, message: ConversationMessage) -> bool:
        """判断消息是否重要（包含技术关键词、项目名或追问链路）"""
        content = message.content.lower()
        # 包含技术关键词
        for keyword in _TECH_KEYWORDS:
            if keyword.lower() in content:
                return True
        # 包含项目名称
        if _PROJECT_PATTERN.search(message.content):
            return True
        # 追问链路
        if _FOLLOWUP_PATTERN.search(message.content):
            return True
        return False

    async def _call_llm(self, prompt: str) -> str:
        """通过 httpx 调用 OpenAI 兼容 API 进行摘要生成"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的面试对话摘要助手。请将以下面试对话压缩为精简摘要，"
                        "保留所有技术要点、项目经验、候选人能力评估等关键信息。"
                        "输出格式为简洁的中文摘要。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data["choices"][0]["message"]["content"]

    async def maybe_compress(
        self, messages: list[ConversationMessage]
    ) -> SummaryResult | None:
        """检查对话长度，超过阈值时触发压缩

        Returns:
            SummaryResult 如果执行了压缩，否则 None
        """
        # 计算总 Token 数
        total_text = "\n".join(m.content for m in messages)
        original_tokens = self._count_tokens(total_text)

        if original_tokens <= self._threshold:
            return None

        logger.info(
            "对话 Token 数 %d 超过阈值 %d，触发压缩",
            original_tokens, self._threshold,
        )

        # 提取关键实体
        entities = self._extract_entities(messages)

        # 分类消息：重要消息 vs 可丢弃消息
        important_messages: list[ConversationMessage] = []
        for msg in messages:
            if self._is_filler(msg):
                continue
            if self._is_important(msg):
                important_messages.append(msg)

        # 如果筛选后仍有过多的消息，保留最近的对话作为补充
        if not important_messages:
            # 兜底：至少保留最近 30% 的消息
            keep_count = max(1, len(messages) // 3)
            important_messages = messages[-keep_count:]

        # 构建摘要 prompt
        conversation_text = "\n".join(
            f"[{m.role}]: {m.content}" for m in important_messages
        )
        entity_hint = "、".join(entities) if entities else "无"
        prompt = (
            f"以下是面试对话的关键内容（已过滤冗余消息），请生成精简摘要：\n\n"
            f"关键实体：{entity_hint}\n\n"
            f"对话内容：\n{conversation_text}\n\n"
            f"请生成一段摘要，保留所有技术评估要点和候选人表现信息。"
        )

        # 调用 LLM 生成摘要
        summary_text = await self._call_llm(prompt)
        compressed_tokens = self._count_tokens(summary_text)

        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0.0

        logger.info(
            "压缩完成: %d -> %d tokens, ratio=%.2f, entities=%d",
            original_tokens, compressed_tokens, compression_ratio, len(entities),
        )

        return SummaryResult(
            summary=summary_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            entities_preserved=entities,
        )
