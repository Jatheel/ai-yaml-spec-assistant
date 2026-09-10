"""Production-grade FastAPI backend with SSE streaming, structured logging, and REST endpoints."""

import json
import time
import uuid
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from agent.agent import doc_agent, run_doc_agent
from agent.schemas import AgentDocResponse
from tools.alternative_docs_tool import (
    _load_yaml,
    _find_badge,
    _format_alternatives,
    _normalize_key,
)
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("doc-agent.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI YAML Specification Assistant API",
    description="Production-grade documentation intelligence engine with LangGraph, Groq, and ChromaDB",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    timestamp: str


class BadgeSummary(BaseModel):
    name: str
    key: str
    mandatory_documents: list[str]
    alternative_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(status_code: int, error: str, detail: str | None = None):
    """Create a structured HTTP error response."""
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            error=error,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Enhanced health check with dependency status."""
    health = {
        "status": "healthy",
        "service": "doc-agent-api",
        "version": "2.0.0",
        "groq_model": settings.groq_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        data = _load_yaml()
        health["yaml_badges_loaded"] = len(data.get("badges", []))
    except Exception as e:
        health["yaml_status"] = f"error: {e}"
        health["status"] = "degraded"

    return health


@app.get("/badges", response_model=list[BadgeSummary])
def list_badges():
    """List all available certification badges in the specification database."""
    try:
        data = _load_yaml()
        badges = data.get("badges", [])
        return [
            BadgeSummary(
                name=b.get("name", "Unknown"),
                key=b.get("key", _normalize_key(str(b.get("name", "")))),
                mandatory_documents=[str(d) for d in b.get("mandatory_documents", [])],
                alternative_count=len(b.get("alternative_groups", [])),
            )
            for b in badges
        ]
    except Exception as e:
        logger.error(f"Badge listing failed: {e}")
        raise _make_error(500, "Failed to load badges", str(e))


@app.get("/badges/{badge_key}")
def get_badge_alternatives(badge_key: str):
    """Get alternative documents for a specific badge by key or name."""
    try:
        data = _load_yaml()
        badge = _find_badge(data["badges"], badge_key)
        if not badge:
            available = [
                b.get("key", _normalize_key(str(b.get("name", ""))))
                for b in data["badges"]
            ]
            raise _make_error(
                404,
                f"Badge '{badge_key}' not found",
                f"Available keys: {', '.join(available)}",
            )
        return {
            "badge": badge.get("name"),
            "key": badge.get("key"),
            "mandatory_documents": badge.get("mandatory_documents", []),
            "alternative_groups": badge.get("alternative_groups", []),
            "formatted": _format_alternatives(badge),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Badge detail failed: {e}")
        raise _make_error(500, "Failed to retrieve badge data", str(e))


@app.post("/query", response_model=AgentDocResponse)
def query_agent(request: QueryRequest):
    """Synchronously executes the documentation agent and returns the structured response."""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"POST /query | session={session_id} | query={request.query[:80]}")
    start = time.time()

    try:
        response = run_doc_agent(request.query, thread_id=session_id)
        if not response:
            raise _make_error(500, "Failed to synthesize structured response.")
        elapsed = round((time.time() - start) * 1000)
        logger.info(
            f"Query done | session={session_id} | {elapsed}ms | "
            f"confidence={response.confidence_score}"
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed | session={session_id} | {e}")
        raise _make_error(500, "Agent execution failed", str(e))


@app.post("/stream")
async def stream_agent(request: QueryRequest):
    """Streams LangGraph node execution events via Server-Sent Events (SSE).

    Event types:
    - start: Agent graph initialized
    - agent_thinking: Agent evaluating context / deciding on tools
    - tool_start: Tool invocation beginning (includes tool name)
    - tool_result: Tool returned data (includes preview)
    - synthesizing: Structured response being generated
    - complete: Final response with timing
    - error: Failure details
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"POST /stream | session={session_id} | query={request.query[:80]}")

    async def event_generator():
        start = time.time()

        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "structured_response": None,
            "iteration_count": 0,
            "retrieval_context": [],
            "tools_invoked": [],
        }
        config = {"configurable": {"thread_id": session_id}}

        yield _sse_event(
            {
                "event": "start",
                "message": "Initializing agent graph...",
                "session_id": session_id,
            }
        )
        await asyncio.sleep(0.05)

        try:
            async for output in doc_agent.astream(initial_state, config=config):
                for node_name, state_update in output.items():
                    if node_name == "agent":
                        msgs = state_update.get("messages", [])
                        tool_calls = []
                        for m in msgs:
                            if hasattr(m, "tool_calls") and m.tool_calls:
                                tool_calls = [
                                    tc.get("name", "unknown")
                                    for tc in m.tool_calls
                                ]

                        if tool_calls:
                            tool_list = ", ".join(tool_calls)
                            yield _sse_event(
                                {
                                    "event": "agent_thinking",
                                    "message": f"Agent invoking tools: {tool_list}",
                                    "tools": tool_calls,
                                }
                            )
                        else:
                            yield _sse_event(
                                {
                                    "event": "agent_thinking",
                                    "message": "Agent evaluating context...",
                                }
                            )

                    elif node_name == "tools":
                        msgs = state_update.get("messages", [])
                        for m in msgs:
                            tool_name = m.name if hasattr(m, "name") else "tool"
                            preview = ""
                            if hasattr(m, "content") and m.content:
                                preview = m.content[:120]
                                if len(m.content) > 120:
                                    preview += "..."
                            yield _sse_event(
                                {
                                    "event": "tool_result",
                                    "tool": tool_name,
                                    "preview": preview,
                                    "message": f"Retrieved data from {tool_name}",
                                }
                            )

                    elif node_name == "synthesize":
                        yield _sse_event(
                            {
                                "event": "synthesizing",
                                "message": "Generating structured response from retrieved sources...",
                            }
                        )
                        await asyncio.sleep(0.1)

                        structured = state_update.get("structured_response")
                        if structured:
                            elapsed = round((time.time() - start) * 1000)
                            yield _sse_event(
                                {
                                    "event": "complete",
                                    "data": structured.model_dump(),
                                    "elapsed_ms": elapsed,
                                    "session_id": session_id,
                                }
                            )
                            logger.info(
                                f"Stream done | session={session_id} | {elapsed}ms"
                            )
        except Exception as err:
            logger.error(f"Stream error | session={session_id} | {err}")
            yield _sse_event({"event": "error", "message": str(err)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/v1/ingest/ide/event")
@app.post("/api/v1/ingest/browser/event")
def silence_telemetry():
    """Dummy endpoints to swallow background IDE/browser telemetry and prevent 404 logs."""
    return {"status": "ok"}