from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
import sys
import os
import re
import yaml
from pathlib import Path

# Load .env
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.alternative_docs_tool import get_alternative_docs

YAML_PATH = Path(__file__).parent.parent / "data" / "alternative_docs.yaml"


def _load_badge_names() -> list[str]:
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [b["name"] for b in data.get("badges", [])]


BADGE_NAMES = _load_badge_names()


def _extract_missing_badge(user_message: str) -> str | None:
    text = user_message.lower()
    missing_signal = (
        "doesn't have" in text
        or "does not have" in text
        or "missing" in text
        or "unavailable" in text
        or "expired" in text
        or "no " in text
    )
    if not missing_signal:
        return None

    cleaned = re.sub(r"\s+certificate\b", "", text).strip()
    for badge in BADGE_NAMES:
        if badge.lower() in cleaned:
            return badge
    return None

# System prompt
SYSTEM_PROMPT = """
You are a certification document verification assistant for a supplier documentation platform.

Your job is to help users when a supplier cannot provide a required certification document.

Rules:
- When a user mentions a missing, expired, or unavailable certification document, 
  identify the badge name and call the get_alternative_docs tool immediately.
- Always present options from highest confidence to lowest.
- Clearly tell the user whether they need ALL documents in a combination or just one.
- Highlight the conditions that must be met across documents.
- Warn about common issues for each option.
- Be conversational, clear, and helpful.
- Never guess alternative documents from memory — always call the tool.

Available specifications include: FSSC 22000, USDA Organic, GLOBALG.A.P., BRCGS, 
ISO 22000, SQF, HACCP, GMP, IFS Food, PrimusGFS, CanadaGAP, WFTO, 
Certified Vegan, Non-GMO Project Verified, Fairtrade International, 
Fair Trade Certified, Certified Gluten-Free, Clean Label Project, 
Certified C.L.E.A.N., Certified Free From Allergens.
"""

MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def _convert_history(history: list) -> list:
    messages = []
    for msg in history or []:
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _polish_tool_response(user_message: str, badge_name: str, tool_output: str) -> str:
    """Rewrite raw tool output into a concise, conversational answer."""
    prompt = (
        "Rewrite the following tool output for a user-friendly chat response.\n"
        "Requirements:\n"
        "- Keep facts identical to the tool output.\n"
        "- Keep option ordering and confidence order.\n"
        "- Be concise and conversational.\n"
        "- Keep critical conditions and common issues.\n"
        "- Do not add disclaimers.\n"
        "- Do not invent fields or options.\n\n"
        f"User request: {user_message}\n"
        f"Badge: {badge_name}\n\n"
        f"Tool output:\n{tool_output}"
    )

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGroq(model=model_name, temperature=0)
            response = llm.invoke([
                SystemMessage(content="You are a helpful certification assistant."),
                HumanMessage(content=prompt),
            ])
            polished = response.content
            lower = polished.lower()
            has_action = (
                "recommended action" in lower
                or "start with option 1" in lower
                or ("option 1" in lower and "recommend" in lower)
            )
            if not has_action:
                polished = (
                    polished.rstrip()
                    + "\n\nRecommended Action:\n"
                    + "Start with Option 1 (highest confidence) and upload all required document(s)."
                )
            return polished
        except Exception as e:
            last_error = e
            err_text = str(e).lower()
            is_rate_limit = "rate limit" in err_text or "429" in err_text or "rate_limit_exceeded" in err_text
            if is_rate_limit:
                continue
            raise

    # Fallback to raw tool output if all polish attempts are rate-limited.
    lower = tool_output.lower()
    has_action = (
        "recommended action" in lower
        or "start with option 1" in lower
        or ("option 1" in lower and "recommend" in lower)
    )
    if not has_action:
        return (
            tool_output.rstrip()
            + "\n\nRecommended Action:\n"
            + "Start with Option 1 (highest confidence) and upload all required document(s)."
        )
    return tool_output

def build_agent(model_name: str):
    llm = ChatGroq(model=model_name, temperature=0)
    tools = [get_alternative_docs]
    agent = create_react_agent(llm, tools)
    return agent


def run_agent(user_message: str, history: list = None) -> str:
    # Keep direct missing-certificate queries deterministic and tool-grounded.
    direct_badge = _extract_missing_badge(user_message)
    if direct_badge:
        raw = get_alternative_docs.invoke({"badge_name": direct_badge})
        return _polish_tool_response(user_message, direct_badge, raw)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if history:
        messages.extend(_convert_history(history))

    messages.append(HumanMessage(content=user_message))

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            agent = build_agent(model_name)
            result = agent.invoke({"messages": messages})

            # Get the last AI message
            for msg in reversed(result["messages"]):
                if msg.__class__.__name__ == "AIMessage":
                    return msg.content

            return "Sorry, I could not generate a response."
        except Exception as e:
            last_error = e
            err_text = str(e).lower()
            is_rate_limit = "rate limit" in err_text or "429" in err_text or "rate_limit_exceeded" in err_text
            if is_rate_limit:
                continue
            raise

    raise RuntimeError(
        "Rate limit reached on available models. Please retry in a few minutes."
    ) from last_error


# Quick terminal test
if __name__ == "__main__":
    print("Agent ready. Type your message (ctrl+c to quit)\n")
    history = []
    while True:
        try:
            user_input = input("You: ")
            response = run_agent(user_input, history)
            print(f"\nAgent: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break