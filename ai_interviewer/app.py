"""AI 模拟面试 Web 应用入口

启动命令：
    python -m ai_interviewer.app

访问地址：
    http://localhost:8000/
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ai_interviewer.agent import InterviewGraphAgent
from ai_interviewer.agent.topic_prioritizer import prioritize_by_jd
from ai_interviewer.jd_engine.jd_loader import get_position, list_positions
from ai_interviewer.resume_parser import parse_pdf, parse_text

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#  全局面试 Agent 实例（API Key 运行时从请求中传入）
# ═══════════════════════════════════════════

_agent = InterviewGraphAgent(api_key="")
_sessions: dict[str, InterviewGraphAgent] = {}

# ══════════════════════════════════════════
#  FastAPI 应用
# ═══════════════════════════════════════════

app = FastAPI(title="AI 模拟面试", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════

def _get_agent(session_id: str) -> InterviewGraphAgent:
    """获取或创建绑定到 session 的 Agent"""
    if session_id not in _sessions:
        _sessions[session_id] = InterviewGraphAgent(api_key="")
    return _sessions[session_id]


def _parse_resume(text: str | None, is_pdf: bool = False) -> dict:
    """解析简历"""
    if not text:
        raise HTTPException(status_code=400, detail="简历内容不能为空")
    if is_pdf:
        # PDF 前端传 bytes base64 的话这里简化处理
        import base64
        data = base64.b64decode(text)
        rd = parse_pdf(data)
    else:
        rd = parse_text(text)
    return {
        "raw_text": rd.raw_text,
        "name": rd.name,
        "skills": rd.skills,
        "projects": rd.projects,
        "experience_years": rd.experience_years,
        "education": rd.education,
        "summary": rd.summary,
    }


# ═══════════════════════════════════════════
#  面试 API
# ═══════════════════════════════════════════

@app.get("/")
async def index():
    """返回前端页面"""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "AI 模拟面试服务运行中"}


@app.get("/api/positions")
async def api_list_positions():
    """返回前端可选的岗位列表（腾讯 JD）

    示例返回：
    [{
      "id": "agent_dev_engineer",
      "name": "Agent 开发工程师",
      "group": "TEG AI 平台部",
      "location": "深圳/北京",
      "summary": "负责 AI Agent 框架与推理服务开发..."
    }]
    """
    try:
        positions = list_positions()
    except Exception as e:
        logger.exception("加载岗位列表失败")
        raise HTTPException(status_code=500, detail=f"加载岗位列表失败: {str(e)}")

    def _desc_text(p) -> str:
        if p.description:
            return "；".join(p.description)
        if p.requirements:
            return "；".join(p.requirements)
        return p.level

    return [
        {
            "id": p.id,
            "name": p.name,
            "group": p.group,
            "location": p.location,
            "summary": _desc_text(p)[:180],
            "requirements_count": len(p.requirements),
            "bonus_count": len(p.bonuses),
        }
        for p in positions
    ]


@app.post("/api/parse-resume")
async def api_parse_resume(request: Request):
    """解析简历（文本格式）"""
    data = await request.json()
    text = data.get("text", "")
    rd = parse_text(text)
    return {
        "raw_text": rd.raw_text,
        "name": rd.name,
        "skills": rd.skills,
        "projects": rd.projects,
        "experience_years": rd.experience_years,
        "education": rd.education,
        "summary": rd.summary,
    }


@app.post("/api/parse-pdf")
async def api_parse_pdf(request: Request):
    """解析 PDF 简历"""
    data = await request.json()
    pdf_base64 = data.get("pdf_data", "")
    if not pdf_base64:
        raise HTTPException(status_code=400, detail="PDF 数据不能为空")
    
    try:
        import base64
        pdf_bytes = base64.b64decode(pdf_base64)
        rd = parse_pdf(pdf_bytes)
        logger.info(f"PDF 解析结果 - 姓名: {rd.name}, 学历: {rd.education}, 年限: {rd.experience_years}, 技能数: {len(rd.skills)}")
        raw_sample = rd.raw_text[:500]
        logger.info(f"PDF 原始文本长度: {len(rd.raw_text)}")
        logger.info(f"PDF 原始文本repr(前200字符): {repr(raw_sample[:200])}")
        return {
            "raw_text": rd.raw_text,
            "name": rd.name,
            "skills": rd.skills,
            "projects": rd.projects,
            "experience_years": rd.experience_years,
            "education": rd.education,
            "summary": rd.summary,
        }
    except Exception as e:
        logger.error("PDF 解析失败: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")


@app.post("/api/start-interview")
async def api_start_interview(request: Request):
    """开始面试（支持前端选择岗位，后端自动注入 JD）

    参数新增 target_position（岗位 ID，可选）：
    - 传了 target_position → 自动从腾讯 JD 里取对应岗位的 name/raw_text 注入到 session state
      plan_interview 节点会基于 JD + 简历动态决定 topic 优先级、话题配额、总题数
    - 没传 target_position → 走通用技术岗 TopicPrioritizer 规则排序（老行为兼容）
    """
    data = await request.json()
    api_key = data.get("api_key", "")
    base_url = data.get("base_url", "https://api.openai.com/v1")
    model = data.get("model", "gpt-4o")
    resume_text = data.get("resume_text", "")
    resume_summary = data.get("resume_summary", resume_text[:1000])
    candidate_name = data.get("candidate_name", "")
    skills = data.get("skills", [])
    target_position = (data.get("target_position") or "").strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="请先配置 API Key")
    if not resume_text:
        raise HTTPException(status_code=400, detail="请先上传简历")

    # ── JD 解析：target_position → name / raw_text ──
    jd_position_name: str = ""
    jd_raw_text: str = ""
    if target_position:
        try:
            pos = get_position(target_position)
            if pos is None:
                raise HTTPException(status_code=400, detail=f"岗位「{target_position}」不存在，请先调用 GET /api/positions 查看可选岗位")
            jd_position_name = pos.name
            jd_raw_text = pos.raw_text
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("JD 解析失败")
            raise HTTPException(status_code=500, detail=f"JD 解析失败: {str(e)}")

    session_id = data.get("session_id") or "session_1"
    agent = InterviewGraphAgent(api_key=api_key, base_url=base_url, model=model)
    _sessions[session_id] = agent

    agent.create_session(
        session_id=session_id,
        resume_text=resume_text,
        resume_summary=resume_summary,
        candidate_name=candidate_name,
        skills=skills,
        target_position=target_position,
        jd_position_name=jd_position_name,
        jd_raw_text=jd_raw_text,
    )

    # ── 首题提速：后台线程预生成面试计划（复用 prioritize_by_jd 结果，不阻塞 start-interview 接口）
    # plan_interview 节点命中 "已有 interview_plan.topics → return {}"，跳过 1 次 LLM，首题快 40%+
    #
    # 为什么放后台线程？因为用户反馈"点开始面试后圈圈转很久"——LLM 调一次要 3~10 秒，
    # 放在 start-interview 里同步跑会让按钮 loading 时间超长（API Key 错/超时甚至能卡 30s）。
    # 先返回 200，后台跑；get_first_question 那边会等结果或抛出真实错误，体验更顺滑。
    session_ref = {"id": session_id, "agent": agent}

    def _bg_pregen_plan():
        try:
            jd_result = prioritize_by_jd(
                api_key=api_key,
                base_url=base_url,
                model=model,
                skills=skills,
                resume_summary=resume_summary,
                jd_position_name=jd_position_name or "通用技术岗",
                jd_raw_text=jd_raw_text,
            )
            sorted_topics = list(jd_result.ordered_topics) or ["技术基础"]
            topic_aliases = {canonical: original for original, canonical in jd_result.ordered_aliases}
            if not topic_aliases and sorted_topics:
                topic_aliases = {t: t for t in sorted_topics}
            topic_question_counter = {t: 0 for t in sorted_topics}
            req_count = sum(1 for v in (jd_result.skill_importance or {}).values() if v == "required")

            interview_plan = {
                "topics": sorted_topics,
                "topic_aliases": topic_aliases,
                "current_topic_index": 0,
                "phase": "deep_dive" if req_count >= 2 else "basics",
                "questions_per_topic": 3,
                "skill_importance": dict(jd_result.skill_importance or {}),
                "topics_per_skill": dict(jd_result.topics_per_skill or {}),
                "topic_question_counter": topic_question_counter,
                "position_id": target_position,
                "position_name": jd_position_name,
                "priority_source": jd_result.source,
            }
            max_questions_final = int(jd_result.suggested_max_questions or 15)
            current_topic_final = sorted_topics[0] if sorted_topics else "技术基础"
            session_ref["agent"]._sessions[session_ref["id"]].update({
                "interview_plan": interview_plan,
                "max_questions": max_questions_final,
                "current_topic": current_topic_final,
                "_plan_ready": True,
                "_plan_error": None,
            })
            logger.info(
                "[start-interview][后台] 预生成面试计划完成(source=%s): position=%s max_q=%d required=%d topics=%s",
                jd_result.source, jd_position_name or "通用", max_questions_final, req_count, sorted_topics[:8]
            )
        except Exception as e:
            logger.exception("[start-interview][后台] 预生成面试计划失败，将在 get_first_question 抛出")
            session_ref["agent"]._sessions.setdefault(session_ref["id"], {}).update({
                "_plan_ready": False,
                "_plan_error": str(e),
            })

    # 先占位：plan 未就绪，get_first_question 会轮询等待 / 抛错
    agent._sessions.setdefault(session_id, {}).update({
        "_plan_ready": False,
        "_plan_error": None,
    })
    threading.Thread(target=_bg_pregen_plan, name=f"plan-{session_id[-6:]}", daemon=True).start()

    return {
        "session_id": session_id,
        "status": "ready",
        "plan_pregenerating": True,
        "position": {
            "id": target_position,
            "name": jd_position_name,
        } if target_position else None,
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    """面试对话（非流式）"""
    data = await request.json()
    session_id = data.get("session_id", "session_1")
    answer = data.get("answer", "")

    agent = _get_agent(session_id)
    if not agent.get_session(session_id):
        raise HTTPException(status_code=404, detail="面试会话不存在，请先调用 /api/start-interview")

    try:
        response = await agent.answer_and_ask(session_id, answer)
    except Exception as e:
        logger.exception("面试对话失败")
        response = f"（系统提示：生成回复时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"

    session = agent.get_session(session_id)
    return {
        "response": response,
        "finished": session.get("is_finished", False) if session else False,
        "question_count": session.get("question_count", 0) if session else 0,
    }


@app.get("/api/plan-status")
async def api_plan_status(session_id: str = "session_1"):
    """轻量轮询接口：返回后台预生成面试计划的进度（避免前端 HTTP 请求卡住）。

    前端 start-interview 之后立刻显示"面试官正在看简历"，
    然后每 0.3s 调这个接口直到 ready=true → 切到"面试官正在输入"再调 first-question。
    """
    agent = _sessions.get(session_id)
    if not agent or not agent.get_session(session_id):
        raise HTTPException(status_code=404, detail="面试会话不存在，请先调用 /api/start-interview")
    s = agent.get_session(session_id) or {}
    plan = s.get("interview_plan") or {}
    return {
        "ready": bool(s.get("_plan_ready")),
        "error": s.get("_plan_error"),   # None 表示没错误
        "position": {
            "id": plan.get("position_id"),
            "name": plan.get("position_name"),
        } if plan.get("position_name") else None,
        "max_questions": s.get("max_questions"),
        "topics_count": len(plan.get("topics", []) or []),
        "required_count": sum(1 for v in (plan.get("skill_importance") or {}).values() if v == "required"),
    }


@app.post("/api/first-question")
async def api_first_question(request: Request):
    """获取第一个面试问题

    等待后台预生成面试计划完成（轮询 最多 45s），ready 了再跑 graph：
    - 如果 plan 预生成失败（_plan_error 非空）直接 400 抛原始错误给前端
    - 如果 plan 预生成好了，graph.plan_interview 命中"已有计划跳过"省 1 次 LLM
    """
    data = await request.json()
    session_id = data.get("session_id", "session_1")

    agent = _get_agent(session_id)
    session = agent.get_session(session_id) or {}

    # ═══ 等后台预生成面试计划 ═══
    if not session.get("_plan_ready"):
        waited = 0.0
        while waited < 45.0:
            # 后台线程失败，直接把真实错误抛前端
            if session.get("_plan_error"):
                msg = str(session["_plan_error"])
                logger.error("首题启动前置失败（后台预生成抛出）: %s", msg)
                raise HTTPException(status_code=400, detail=msg)
            if session.get("_plan_ready"):
                break
            # async sleep，不阻塞 event loop
            await asyncio.sleep(0.3)
            waited += 0.3
            # 刷新 session 引用（后台线程 update 了 dict）
            session = agent.get_session(session_id) or {}
        # 超时兜底
        if not session.get("_plan_ready") and not session.get("_plan_error"):
            err = (
                f"面试计划预生成超时（45s）。可能原因：model={agent.model!r} 或 base_url={agent.base_url!r} 连不上，"
                f"或 API Key 响应慢；请检查配置后重试。"
            )
            logger.error(err)
            raise HTTPException(status_code=408, detail=err)
        if session.get("_plan_error"):
            raise HTTPException(status_code=400, detail=str(session["_plan_error"]))

    try:
        response = await agent.get_first_question(session_id)
    except RuntimeError as e:
        msg = str(e) or ""
        # JD 排序失败、LLM 连不上等"启动前置条件不满足"一律 400 抛给前端弹窗
        if "JD 话题排序失败" in msg or "面试无法开始" in msg:
            logger.error("首题启动前置失败: %s", msg)
            raise HTTPException(status_code=400, detail=msg) from e
        # 其他 RuntimeError 走兜底字符串（理论上不会走到）
        logger.exception("获取首题 RuntimeError")
        response = f"（系统提示：生成问题时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"
    except Exception as e:
        logger.exception("获取首题失败")
        response = f"（系统提示：生成问题时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"

    session = agent.get_session(session_id)
    return {
        "response": response,
        "finished": session.get("is_finished", False) if session else False,
        "question_count": session.get("question_count", 0) if session else 0,
    }


async def _stream_response(agent: InterviewGraphAgent, session_id: str, answer: str):
    """流式返回面试官回复（出错时以 error chunk 返回，避免 SSE 永远阻塞）"""
    try:
        response = await agent.answer_and_ask(session_id, answer)
    except Exception as e:
        logger.exception("流式生成回复失败")
        error_msg = f"（系统提示：生成回复时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"
        yield f"data: {json.dumps({'content': error_msg, 'finished': False, 'error': True}, ensure_ascii=False)}\n\n"
        session = agent.get_session(session_id)
        yield f"data: {json.dumps({'content': '', 'finished': session.get('is_finished', False) if session else False, 'question_count': session.get('question_count', 0) if session else 0}, ensure_ascii=False)}\n\n"
        return

    session = agent.get_session(session_id)

    chunk_size = 8
    for i in range(0, len(response), chunk_size):
        chunk = response[i:i + chunk_size]
        yield f"data: {json.dumps({'content': chunk, 'finished': False}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.03)

    yield f"data: {json.dumps({'content': '', 'finished': session.get('is_finished', False) if session else False, 'question_count': session.get('question_count', 0) if session else 0}, ensure_ascii=False)}\n\n"


@app.post("/api/chat-stream")
async def api_chat_stream(request: Request):
    """面试对话（SSE 流式）"""
    data = await request.json()
    session_id = data.get("session_id", "session_1")
    answer = data.get("answer", "")

    agent = _get_agent(session_id)
    if not agent.get_session(session_id):
        raise HTTPException(status_code=404, detail="面试会话不存在")

    return StreamingResponse(
        _stream_response(agent, session_id, answer),
        media_type="text/event-stream",
    )


@app.get("/api/evaluation")
async def api_evaluation(session_id: str = "session_1"):
    """获取面试评估"""
    agent = _get_agent(session_id)
    evaluation = await agent.get_evaluation(session_id)
    return evaluation


@app.get("/api/health")
async def api_health():
    """健康检查"""
    return {"status": "ok"}


@app.get("/api/agent-state")
async def api_agent_state(session_id: str = "session_1"):
    """查看 LangGraph Agent 内部状态（Planning + Memory）"""
    agent = _get_agent(session_id)
    state = agent.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "candidate_profile": state.get("candidate_profile", {}),
        "interview_plan": state.get("interview_plan", {}),
        "current_topic": state.get("current_topic", ""),
        "question_count": state.get("question_count", 0),
        "max_questions": state.get("max_questions", 15),
        "is_finished": state.get("is_finished", False),
        "response_quality": state.get("response_quality", ""),
    }


# ═══════════════════════════════════════════
#  RAG 知识库 API
# ═══════════════════════════════════════════

@app.get("/api/knowledge-base")
async def api_knowledge_base_overview():
    """知识库概览"""
    from ai_interviewer.rag_engine.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    if not kb.is_ready:
        await kb.load()
    return kb.to_dict()


@app.get("/api/knowledge-base/categories")
async def api_knowledge_base_categories():
    """获取知识库分类"""
    from ai_interviewer.rag_engine.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    if not kb.is_ready:
        await kb.load()
    return {"categories": kb.get_categories()}


@app.get("/api/knowledge-base/search")
async def api_knowledge_base_search(q: str = "", top_k: int = 5):
    """搜索知识库"""
    from ai_interviewer.rag_engine.retriever import get_retriever
    retriever = get_retriever()
    result = await retriever.retrieve(q, top_k=top_k)
    return {
        "query": result.query,
        "items": [item.to_dict() for item in result.items],
        "scores": result.scores,
    }


# ═══════════════════════════════════════════
#  静态文件挂载
# ═══════════════════════════════════════════

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ═══════════════════════════════════════════
#  启动入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    uvicorn.run(app, host=host, port=port)
