# api/main.py

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import yaml
from pathlib import Path
import re

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import run_agent

app = FastAPI(title="AI YAML Specification Assistant API")

# Allow frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YAML_PATH = Path(__file__).parent.parent / "data" / "alternative_docs.yaml"


def _load_badges_data() -> dict:
    # Use utf-8-sig to safely handle files saved with BOM.
    with open(YAML_PATH, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _find_badge(badges: list[dict], badge_input: str) -> dict | None:
    input_key = _normalize_key(badge_input)

    badge = next((b for b in badges if _normalize_key(str(b.get("key", ""))) == input_key), None)
    if badge:
        return badge

    badge = next((b for b in badges if str(b.get("name", "")).strip().lower() == badge_input.strip().lower()), None)
    if badge:
        return badge

    return next((b for b in badges if badge_input.strip().lower() in str(b.get("name", "")).strip().lower()), None)

# ---- Request/Response models ----

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ChatResponse(BaseModel):
    response: str

# ---- Routes ----

@app.get("/")
def root():
    return {"status": "AI YAML Specification Assistant API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint — takes a user message and returns agent response."""
    try:
        response = run_agent(request.message, request.history)
        return ChatResponse(response=response)
    except RuntimeError as e:
        msg = str(e)
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail=msg)
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(status_code=429, detail="Rate limit reached. Please retry in a few minutes.")
        raise HTTPException(status_code=500, detail=msg)


@app.get("/badges")
def list_badges():
    """Returns all available badge names."""
    try:
        data = _load_badges_data()
        badges = [b["name"] for b in data["badges"]]
        return {"badges": badges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alternatives/{badge_name}")
def get_alternatives(badge_name: str):
    """Returns alternative document options for a specific badge."""
    try:
        data = _load_badges_data()
        badges = data.get("badges", [])
        badge = _find_badge(badges, badge_name)

        if not badge:
            raise HTTPException(status_code=404, detail=f"Badge '{badge_name}' not found")

        return {
            "badge": badge["name"],
            "mandatory_documents": badge["mandatory_documents"],
            "alternative_groups": badge["alternative_groups"],
            "available_alternative_documents": badge["available_alternative_documents"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))