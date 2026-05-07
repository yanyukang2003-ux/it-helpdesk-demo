"""
API 服务
提供 HTTP 接口供前端或企业 IM 调用

流程引擎：动态 plan-and-execute（参见 app/dynamic.py）。
"""

import os
import uuid
import json
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import config
from app.dynamic import (
    stream_run,
    stream_branch,
    list_run_steps,
    parse_cp_id,
    RUNS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 启动 Helpdesk Agent (dynamic plan-and-execute)")
    yield
    print("👋 Agent 关闭")


app = FastAPI(
    title="IT Helpdesk Agent API",
    description="动态步骤规划 + 可分支重跑的问答 Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., examples=["VPN 怎么连接？"])
    user_id: str = "anonymous"
    user_name: str = ""
    user_department: str = ""
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class PredictRequest(BaseModel):
    partial: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "it-helpdesk-agent"


class BranchRequest(BaseModel):
    """从某个 step checkpoint 分叉重跑。

    checkpoint_id 形如 ``<thread_id>:<step_index>``。

    override_state 可包含：
      - primary: 替换该步的主输出文本（重跑后续步骤）
      - alt_idx: 配合 primary，标记选中的是第几个 alternative
      - style_hint: 仅对 answer 类型步骤有效，用风格重新生成回答
    """
    source_thread_id: str
    from_checkpoint_id: str = ""
    new_human_content: str | None = None
    override_state: dict | None = None
    as_new_thread: bool = False


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """根据已输入的部分文本预测可能的完整问题。"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.ROUTER_MODEL, temperature=0.3)
    prompt = f"""用户正在输入一个问题，目前已输入：
"{req.partial}"

请预测最可能的 3 个完整问题。要求：
1. 每行一个问题，不要编号、不要前缀
2. 自然完整
3. 基于已输入内容合理推断
4. 如果已输入已经是完整问题，给出 3 个相关后续问题"""
    try:
        response = llm.invoke(prompt)
        lines = [l.strip() for l in str(response.content).strip().split("\n") if l.strip()]
        cleaned = [re.sub(r'^[\d]+[\.\)\、\s]+', '', l).strip() for l in lines[:3]]
        return PredictResponse(predictions=cleaned[:3])
    except Exception:
        return PredictResponse(predictions=[f"{req.partial.strip()}？"])


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """非流式对话；内部完整跑完动态流程后返回最终回答。"""
    thread_id = req.thread_id or str(uuid.uuid4())
    reply = ""
    try:
        for ev in stream_run(thread_id, req.message):
            if ev.get("type") == "done":
                reply = ev.get("reply", "")
        if not reply:
            reply = "抱歉，我暂时无法处理你的请求，请稍后再试。"
        return ChatResponse(reply=reply, thread_id=thread_id)
    except Exception as e:
        print(f"❌ 处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式：plan_complete → step_complete × N → done"""
    thread_id = req.thread_id or str(uuid.uuid4())

    async def event_generator():
        try:
            for ev in stream_run(thread_id, req.message):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"❌ stream 失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/threads/{thread_id}/checkpoints")
async def get_thread_checkpoints(thread_id: str, limit: int = Query(100, ge=1, le=500)):
    """返回某 thread 的步骤列表（每步即一个 checkpoint）。"""
    rows = list_run_steps(thread_id)
    return {"thread_id": thread_id, "checkpoints": rows[:limit]}


@app.post("/threads/branch")
async def branch_endpoint(req: BranchRequest):
    """从某 step 分叉重跑（流式返回新事件）。"""
    if req.source_thread_id not in RUNS:
        raise HTTPException(status_code=400, detail="不存在的会话或会话已过期")

    # Parse checkpoint -> step index
    from_index = -1
    if req.from_checkpoint_id:
        try:
            tid, idx = parse_cp_id(req.from_checkpoint_id)
            if tid != req.source_thread_id:
                raise HTTPException(status_code=400, detail="checkpoint 与 thread 不匹配")
            from_index = idx
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    override = req.override_state or {}
    override_primary = override.get("primary")
    style_hint = override.get("style_hint")
    alt_idx = override.get("alt_idx")

    async def event_generator():
        try:
            for ev in stream_branch(
                req.source_thread_id,
                from_index,
                override_primary=override_primary,
                override_alt_idx=alt_idx,
                style_hint=style_hint,
                override_user_message=req.new_human_content,
                as_new_thread=req.as_new_thread,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            print(f"❌ branch 失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------
# Frontend static
# --------------------------------------------------

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=True)
