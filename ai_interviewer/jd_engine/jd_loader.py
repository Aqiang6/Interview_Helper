"""岗位 JD 加载与解析引擎 — 基于腾讯校招 JD 文本。

结构化数据供：
1. 前端选岗位的下拉列表（/api/positions）
2. LLM 动态生成话题优先级（根据不同岗位 JD 的要求强度）
3. 路由决策：回答质量 × JD 要求强度的二维决策表
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JobPosition:
    """单个岗位 JD 的结构化视图。"""

    position_id: str                      # 英文 id，如 "agent_dev_engineer"
    name: str                             # 中文岗位名，如 "Agent开发工程师"
    level: str                            # 职级（应届毕业生/社招等）
    # 前端下拉展示需要的展示字段（腾讯校招 JD 原文没拆分，做合理默认值；如需自定义可在 parse 时覆盖）
    group: str = "TEG AI 平台部"            # 所属 BG/部门
    location: str = "深圳 / 北京"            # 工作地点
    description: list[str] = field(default_factory=list)   # 岗位描述（分条）
    requirements: list[str] = field(default_factory=list)  # 岗位要求（分条）
    bonuses: list[str] = field(default_factory=list)       # 加分项（分条）
    raw_text: str = ""                                        # JD 完整原文（给 LLM 读）

    @property
    def id(self) -> str:
        """对外接口统一用 id 字段（position_id 的别名，前端习惯取 id）。"""
        return self.position_id

    @property
    def short_tagline(self) -> str:
        """前端下拉列表展示的简短描述（取岗位要求第1条）。"""
        if self.requirements:
            head = self.requirements[0]
            return (head[:36] + "…") if len(head) > 38 else head
        return self.level

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "name": self.name,
            "level": self.level,
            "short_tagline": self.short_tagline,
            "sections": {
                "description": self.description,
                "requirements": self.requirements,
                "bonuses": self.bonuses,
            },
        }


# JD 文件路径（与本模块同目录或 project 根）
_DEFAULT_JD_PATH = Path(__file__).resolve().parent.parent / "腾讯jd.txt"

# 岗位名 → position_id 映射（解析时用）
# 注意：对外接口前端用的 id 保持语义清晰；同时保留别名避免传错。
_NAME_TO_ID = {
    "ai全栈工程师": "ai_fullstack_engineer",
    "全栈ai工程师": "ai_fullstack_engineer",
    "agent开发工程师": "agent_dev_engineer",
    "agent工程师": "agent_dev_engineer",
    "ai应用工程师": "ai_application_engineer",
    "后台开发": "backend_engineer",
    "后端开发": "backend_engineer",
    "后台开发工程师": "backend_engineer",
}


def _slug_to_id(name: str) -> str:
    key = name.strip().lower()
    if key in _NAME_TO_ID:
        return _NAME_TO_ID[key]
    # 兜底：中文拼音式 slug
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return safe or "unknown_position"


def _parse_jd_text(text: str) -> list[JobPosition]:
    """解析 JD 原文 → 结构化岗位列表。"""
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.splitlines()]

    # 定位每个岗位的起始行：下一行是 "技术"，再下一行是"应届毕业生" → 这就是岗位头
    positions: list[JobPosition] = []
    starts: list[int] = []
    for i in range(len(lines) - 2):
        if (
            lines[i].strip()
            and lines[i + 1].strip() == "技术"
            and "届毕业生" in lines[i + 2]
        ):
            starts.append(i)

    # 每段切片
    for seg_idx, start in enumerate(starts):
        end = starts[seg_idx + 1] if seg_idx + 1 < len(starts) else len(lines)
        block = lines[start:end]
        positions.append(_parse_single_block(block))

    return [p for p in positions if p.name]


def _parse_single_block(block: list[str]) -> JobPosition:
    """解析单个岗位的文本块。"""
    pos = JobPosition(position_id="", name="", level="")
    pos.name = block[0].strip()
    # block[1] = "技术"
    pos.level = block[2].strip() if len(block) > 2 else "应届毕业生"
    pos.position_id = _slug_to_id(pos.name)

    # 分 section：岗位描述 / 岗位要求 / 加分项或注意事项
    section_name = None
    numbered_re = re.compile(r"^\s*\d+[、.．]\s*(.+)$")

    for line in block[3:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "岗位描述":
            section_name = "description"
            continue
        if stripped == "岗位要求":
            section_name = "requirements"
            continue
        if "加分项" in stripped or "注意事项" in stripped:
            section_name = "bonuses"
            continue
        m = numbered_re.match(stripped)
        if m and section_name:
            item = m.group(1).strip()
            if section_name == "description":
                pos.description.append(item)
            elif section_name == "requirements":
                pos.requirements.append(item)
            elif section_name == "bonuses":
                pos.bonuses.append(item)
        elif section_name:
            # 没有编号但属于当前 section 的头行（岗位描述里的"你是腾讯产品背后的英雄"这种就是）
            target = {
                "description": pos.description,
                "requirements": pos.requirements,
                "bonuses": pos.bonuses,
            }[section_name]
            if target:
                # 追加到上一条末尾（作为解释性语句）
                target[-1] = target[-1] + " " + stripped
            else:
                target.append(stripped)

    # raw_text：岗位块完整原文（留着 LLM 深度消费）
    pos.raw_text = "\n".join(b for b in block if b.strip())
    return pos


# ═══════════════════════════════════════════
#  全局 Loader（单例 + 懒加载）
# ═══════════════════════════════════════════

_positions_cache: list[JobPosition] | None = None


def load_jobs(path: str | Path | None = None, force_reload: bool = False) -> list[JobPosition]:
    """加载并解析 JD 文件。默认读 ai_interviewer/腾讯jd.txt。"""
    global _positions_cache
    if _positions_cache is None or force_reload:
        jd_path = Path(path) if path else _DEFAULT_JD_PATH
        if not jd_path.exists():
            logger.warning("JD 文件不存在: %s", jd_path)
            _positions_cache = []
            return []
        text = jd_path.read_text(encoding="utf-8", errors="ignore")
        _positions_cache = _parse_jd_text(text)
        logger.info("加载 JD 成功: %d 个岗位 [%s]", len(_positions_cache),
                    ", ".join(p.name for p in _positions_cache))
    return _positions_cache


def list_positions() -> list[JobPosition]:
    """返回岗位对象列表（前端选岗位 / 后端按 id 查找都统一用 JobPosition）。"""
    return list(load_jobs())


def get_position(position_id_or_name: str) -> JobPosition | None:
    """按 position_id 或中文名模糊匹配单个岗位。"""
    if not position_id_or_name:
        return None
    jobs = load_jobs()
    key = position_id_or_name.strip().lower()
    for j in jobs:
        if j.position_id.lower() == key or j.name.lower() == key:
            return j
    # 模糊：id/name 包含关键词
    for j in jobs:
        if key in j.position_id.lower() or key in j.name.lower():
            logger.info("模糊匹配岗位: %s -> %s", position_id_or_name, j.name)
            return j
    return None


# ═══════════════════════════════════════════
#  LLM 驱动优先级排序 Prompt（JD-aware）
# ═══════════════════════════════════════════

PRIORITY_SYSTEM_PROMPT = """你是面试话题规划专家。根据目标岗位的 JD（岗位描述+要求+加分项）、候选人简历技能列表、候选人简历摘要，

生成【按面试优先级从高到低排序的话题列表】，并标注每个话题在当前 JD 中的要求强度。

输出必须是严格的 JSON，格式：
```json
{
  "ordered_topics": [
    "RAG系统",
    "Agent架构设计",
    "LangGraph框架实践",
    "Python工程能力",
    "分布式系统基础",
    "Prompt Engineering",
    "向量数据库与检索",
    "全栈开发基础"
  ],
  "skill_importance": {
    "RAG系统": "required",
    "Agent架构设计": "required",
    "LangGraph框架实践": "preferred",
    "Python工程能力": "required",
    "分布式系统基础": "preferred",
    "Prompt Engineering": "preferred",
    "向量数据库与检索": "preferred",
    "全栈开发基础": "preferred"
  },
  "suggested_max_questions": 18,
  "topics_per_skill": {
    "required": 4,
    "preferred": 3,
    "bonus": 2
  }
}
```

### 规则（必须严格遵守）
1. **ordered_topics 只从候选人技能里挑**，不能凭空生成候选人根本没提到过的技能（但允许用 JD 里的对应标准名替代 resume 里的别名，比如简历里写"检索增强"→ 标准名 "RAG系统"）
2. **面试优先级排序依据：**
   - 第1层：JD 岗位要求里明确写的 required 技能 → 最优先问；JD 加分项里的 bonus 技能 → 最末
   - 第2层：同一强度档里，简历有项目经验/线上经验的先问（简历摘要里高频出现的先）
   - 第3层：同大类话题聚类在一起（比如"Agent框架"相关的都排一段，别东一个西一个）
3. **skill_importance 取值只能是 "required" / "preferred" / "bonus"** 三选一：
   - required：岗位要求里明确列的核心技能（不会就不适合这个岗位）
   - preferred：JD 里提到的"了解/熟悉"或偏工程实践的能力（加分但不致命）
   - bonus：只在加分项里出现的、锦上添花的点
4. **suggested_max_questions** 根据 JD 强度决定：
   - Agent工程师 / AI全栈 这种要求技术密度高的岗位：17-20 题
   - AI应用工程师 / 后台开发：13-16 题
5. **topics_per_skill**：同一强度档每题建议问几轮深度追问
   - required：4 轮（追问深一点，这是核心指标）
   - preferred：3 轮
   - bonus：2 轮（点到为止）

只输出 JSON，不要任何其他文字、解释、markdown 代码块标记。"""
