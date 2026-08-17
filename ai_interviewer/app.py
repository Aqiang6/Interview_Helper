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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ai_interviewer.agent import InterviewGraphAgent
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
    """开始面试"""
    data = await request.json()
    api_key = data.get("api_key", "")
    base_url = data.get("base_url", "https://api.openai.com/v1")
    model = data.get("model", "gpt-4o")
    resume_text = data.get("resume_text", "")
    resume_summary = data.get("resume_summary", resume_text[:1000])
    candidate_name = data.get("candidate_name", "")
    skills = data.get("skills", [])

    if not api_key:
        raise HTTPException(status_code=400, detail="请先配置 API Key")
    if not resume_text:
        raise HTTPException(status_code=400, detail="请先上传简历")

    session_id = data.get("session_id") or "session_1"
    agent = InterviewGraphAgent(api_key=api_key, base_url=base_url, model=model)
    _sessions[session_id] = agent

    agent.create_session(
        session_id=session_id,
        resume_text=resume_text,
        resume_summary=resume_summary,
        candidate_name=candidate_name,
        skills=skills,
    )

    return {"session_id": session_id, "status": "ready"}


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


@app.post("/api/first-question")
async def api_first_question(request: Request):
    """获取第一个面试问题"""
    data = await request.json()
    session_id = data.get("session_id", "session_1")

    agent = _get_agent(session_id)
    try:
        response = await agent.get_first_question(session_id)
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
    """流式返回面试官回复"""
    response = await agent.answer_and_ask(session_id, answer)
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
