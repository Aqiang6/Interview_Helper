"""全局配置管理 - 通过环境变量和 .env 文件加载所有配置项"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用全局配置，所有敏感信息均从环境变量加载，禁止硬编码"""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider 配置 ──
    openai_api_key: str = Field(default="", description="主模型 API Key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="主模型 Base URL")
    openai_model: str = Field(default="gpt-4o", description="主模型名称")

    fallback_api_key: str = Field(default="", description="备用模型 API Key")
    fallback_base_url: str = Field(default="https://api.openai.com/v1", description="备用模型 Base URL")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="备用模型名称")

    # ── 加密密钥 ──
    encryption_key: str = Field(default="", description="Fernet 对称加密密钥，为空时自动生成")

    # ── 语义缓存 ──
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding 模型名称",
    )
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0, description="语义缓存命中阈值")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="缓存 TTL（秒）")
    cache_max_size: int = Field(default=10000, ge=100, description="缓存最大条目数")

    # ── RAG 知识库 ──
    rag_enabled: bool = Field(default=True, description="是否启用 RAG 知识库检索")
    rag_top_k: int = Field(default=3, ge=1, le=10, description="RAG 检索返回的条目数")
    rag_min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="RAG 检索最低相似度阈值")

    # ── LangGraph Agent ──
    agent_max_questions: int = Field(default=15, ge=3, le=50, description="单次面试最大问题数")
    agent_recursion_limit: int = Field(default=10, ge=3, le=25, description="LangGraph 递归限制")
    agent_enable_tools: bool = Field(default=True, description="是否启用 Tool Use")

    # ── OpenTelemetry ──
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="ai-agent-interviewer")

    # ── 服务配置 ──
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """获取全局配置单例"""
    return AppSettings()
