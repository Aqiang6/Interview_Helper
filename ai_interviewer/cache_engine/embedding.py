"""Embedding 服务（极简版 - 保持原 sentence-transformers 方案）

用户要求：就用原来的 sentence-transformers，不要用 OpenAI 兼容 API 做 embedding。
核心修复：**跳过 HuggingFace 的检查更新步骤**。
    HuggingFace 每次 from_pretrained 默认先发 HTTPS HEAD 请求检查远端文件是否比缓存新
    → 国内 99% 连 huggingface.co 会 WinError 10060。

    解决办法是：**先 local_files_only=True 直接用本地缓存加载（完全不联网、跳过 HEAD）**
    → 没命中再开联网（并设置 HF_ENDPOINT=国内镜像 hf-mirror.com）下载一次
    → 真下不到才返回 mock 384d 向量，保证面试不中断。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from ai_interviewer.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MOCK_EMBED_DIM = 384  # 与 all-MiniLM-L6-v2 一致


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_name: str = get_settings().embedding_model
        self._lock = asyncio.Lock()
        self._load_errored: bool = False  # 真失败过就不再重试，避免每次都等 30s
        # 环境变量先设好（镜像 + 离线标志 + 遥测关）
        self._preset_env()

    # ------------------------------------------------------------------
    # 基础：HuggingFace 环境变量预设
    # ------------------------------------------------------------------
    @staticmethod
    def _preset_env() -> None:
        # ① 镜像：国内默认用 hf-mirror.com
        if not os.environ.get("HF_ENDPOINT"):
            try:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            except Exception:
                pass
        # ② 禁止 huggingface_hub 遥测（减少一次 HTTP HEAD）
        if not os.environ.get("HF_HUB_DISABLE_TELEMETRY"):
            try:
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            except Exception:
                pass
        # ③ 关闭进度条环境变量（sentence-transformers 有些路径会读）
        if not os.environ.get("DISABLE_PROGRESS_BAR"):
            try:
                os.environ.setdefault("DISABLE_PROGRESS_BAR", "1")
            except Exception:
                pass
        # ④ 禁用全局 tqdm 进度条（去掉 Loading weights 100%|███| 103/103 … 每次启动刷屏）
        #    Loading weights 来自 sentence-transformers/transformers 加载 torch/safetensors 权重时的内部 tqdm，
        #    与 encode(show_progress_bar=...) 无关；设 TQDM_DISABLE=1 是最省心的全局关闭方式。
        if not os.environ.get("TQDM_DISABLE"):
            try:
                os.environ["TQDM_DISABLE"] = "1"
            except Exception:
                pass
        # ⑤ transformers 日志降到 error（关闭 model weights, missing keys 等 warning 刷屏）
        if not os.environ.get("TRANSFORMERS_VERBOSITY"):
            try:
                os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            except Exception:
                pass
        # ⑥ safetensors/transformers 加载权重时不打印 "Some weights of ... not used" 类提示
        if not os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"):
            try:
                os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            except Exception:
                pass
        # ⑦ 真正生效：如果 transformers/huggingface_hub/tqdm 已经被其他地方 import 了，
        #    上面 set env 就来不及了；这里再主动把 tqdm 全局 disable + transformers 日志降级。
        try:  # pragma: no cover - 依赖是否安装
            import tqdm as _tqdm_mod

            _tqdm_mod.tqdm.global_disable = True
        except Exception:
            pass
        try:  # pragma: no cover
            from transformers.utils import logging as _hf_logging

            _hf_logging.set_verbosity_error()
            _hf_logging.disable_progress_bar()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 加载策略：① local_files_only 先命中缓存(跳过 HEAD 检查更新) → ② 联网镜像下载 → ③ None
    # ------------------------------------------------------------------
    def _try_load_local_only(self) -> "SentenceTransformer | None":
        """用本地缓存加载，完全不发任何网络请求（跳过 HuggingFace HEAD 检查更新）。"""
        from sentence_transformers import SentenceTransformer

        logger.info("[Embedding] 尝试本地缓存加载 (local_files_only=True, 不联网)")
        try:
            model = SentenceTransformer(self._model_name, local_files_only=True)
            logger.info("[Embedding] 本地缓存加载成功 ✓（零网络请求）")
            return model
        except Exception as e:  # noqa: BLE001
            # Local cache miss → 正常情况；打 warning 然后走联网
            logger.warning("[Embedding] 本地没缓存或加载失败: %s → 再尝试联网下载一次", e)
            return None

    def _try_load_with_network(self) -> "SentenceTransformer | None":
        from sentence_transformers import SentenceTransformer

        logger.info("[Embedding] 尝试联网下载（镜像 HF_ENDPOINT=%s）", os.environ.get("HF_ENDPOINT"))
        try:
            model = SentenceTransformer(self._model_name)  # 允许联网
            logger.info("[Embedding] 联网下载/检查成功 ✓")
            return model
        except Exception as e:  # noqa: BLE001
            logger.error("[Embedding] 联网仍失败: %s → 降级 mock 384d 向量兜底（不中断面试）", e)
            return None

    async def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_errored:
            return False

        async with self._lock:
            if self._model is not None:
                return True
            if self._load_errored:
                return False

            # ① 先用本地缓存，跳过 HEAD 检查更新（核心！）
            model = await asyncio.to_thread(self._try_load_local_only)
            if model is not None:
                self._model = model
                return True

            # ② 缓存没命中 → 联网下载一次
            model = await asyncio.to_thread(self._try_load_with_network)
            if model is not None:
                self._model = model
                return True

            # ③ 都失败了 → 标记，下次不再重试
            self._load_errored = True
            return False

    # ------------------------------------------------------------------
    # Mock 兜底（永远可用）
    # ------------------------------------------------------------------
    @staticmethod
    def _mock_embed_single(text: str) -> np.ndarray:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(MOCK_EMBED_DIM).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return vec

    @classmethod
    def _mock_embed_batch(cls, texts: list[str]) -> list[np.ndarray]:
        return [cls._mock_embed_single(t) for t in texts]

    # ------------------------------------------------------------------
    # 真实 embed（sentence-transformers）包装
    # ------------------------------------------------------------------
    def _real_embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        assert self._model is not None
        arr: np.ndarray = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [row.astype(np.float32) for row in arr]

    # ------------------------------------------------------------------
    # Public API（保持原形状一致：async embed / embed_batch / compute_similarity）
    # ------------------------------------------------------------------
    async def embed(self, text: str) -> np.ndarray:
        ok = await self._ensure_model()
        if ok:
            try:
                vectors = await asyncio.to_thread(self._real_embed_batch, [text])
                return vectors[0]
            except Exception as e:  # noqa: BLE001
                logger.warning("[Embedding] sentence-transformers 嵌入失败，降级 mock: %s", e)
        return self._mock_embed_single(text)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        ok = await self._ensure_model()
        if ok:
            try:
                return await asyncio.to_thread(self._real_embed_batch, texts)
            except Exception as e:  # noqa: BLE001
                logger.warning("[Embedding] sentence-transformers 批量嵌入失败，降级 mock: %s", e)
        return self._mock_embed_batch(texts)

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
