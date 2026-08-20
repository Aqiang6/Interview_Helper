"""岗位 JD 引擎：解析腾讯校招 JD，供前端选岗位 + LLM 动态生成面试优先级。"""

from ai_interviewer.jd_engine.jd_loader import (
    JobPosition,
    PRIORITY_SYSTEM_PROMPT,
    get_position,
    list_positions,
    load_jobs,
)

__all__ = [
    "JobPosition",
    "PRIORITY_SYSTEM_PROMPT",
    "load_jobs",
    "list_positions",
    "get_position",
]
