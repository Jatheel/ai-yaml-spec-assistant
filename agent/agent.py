"""Production-grade LangGraph documentation intelligence agent.

Features:
- Cyclic tool execution with iteration guards
- Context-isolated synthesis (hallucination mitigation)
- Grounded confidence computation from retrieval signals
- Conversation memory via MemorySaver checkpointer
"""

import re
import logging
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config import settings
from tools.alternative_docs_tool import get_alternative_docs
from tools.semantic_search_tool import semantic_docs_search
from tools.list_badges_tool import list_available_badges
from agent.schemas import AgentDocResponse, RetrievalSource

logger = logging.getLogger("doc-agent.agent")


# ---------------------------------------------------------------------------
# 1. State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Rich graph state with cycle tracking and retrieval context isolation."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    structured_response: AgentDocResponse | None
    iteration_count: int
    retrieval_context: list[str]
    tools_invoked: list[str]


# ---------------------------------------------------------------------------
# 2. Initialize Tools & LLM
# ---------------------------------------------------------------------------

tools = [get_alternative_docs, semantic_docs_search, list_available_badges]
tool_node = ToolNode(tools)

llm = ChatGroq(
    model=settings.groq_model,
    groq_api_key=settings.groq_api_key,
    temperature=settings.llm_temperature,
)
model_with_tools = llm.bind_tools(tools)
model_with_structured_output = llm.with_structured_output(AgentDocResponse)

SYSTEM_PROMPT = (
    "You are a production-grade documentation intelligence agent.\n\n"
    "WORKFLOW:\n"
    "1. ALWAYS use the `get_alternative_docs` tool first when the user mentions "
    "a specific badge or certification name.\n"
    "2. Use `semantic_docs_search` when the query is ambiguous or you need to "
    "discover which badges are relevant.\n"
    "3. Use `list_available_badges` to show the user what certifications are available.\n"
    "4. You may call tools multiple times with refined queries.\n"
    "5. NEVER fabricate specifications — if tools return no data, say so.\n\n"
    "RULES:\n"
    "- Cite the specific badge names and document options from tool output.\n"
    "- Include error recovery guidance when available.\n"
    "- If a badge is not found, list what IS available.\n"
    "- Respond conversationally while maintaining factual accuracy.\n"
    "- Reference previous conversation context when relevant."
)


# ---------------------------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------------------------

def call_model(state: AgentState) -> dict:
    """Processes message history and determines if tool invocation is necessary."""
    messages = list(state["messages"])

    # Inject system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = model_with_tools.invoke(messages)

    iteration = state.get("iteration_count", 0) + 1
    logger.info(f"Agent iteration {iteration}: {'tool_calls' if hasattr(response, 'tool_calls') and response.tool_calls else 'no tools'}")

    return {
        "messages": [response],
        "iteration_count": iteration,
    }


def should_continue(state: AgentState) -> str:
    """Routes to tools or synthesis, with iteration cap to prevent infinite loops."""
    iteration = state.get("iteration_count", 0)

    if iteration >= settings.max_agent_iterations:
        logger.warning(
            f"Agent hit iteration cap ({settings.max_agent_iterations}), forcing synthesis"
        )
        return "synthesize"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesize"


# ---------------------------------------------------------------------------
# 4. Retrieval Context Extraction & Confidence Computation
# ---------------------------------------------------------------------------

def _extract_retrieval_context(
    state: AgentState,
) -> tuple[list[str], list[str], list[RetrievalSource]]:
    """Extract tool outputs, tool names, and build provenance sources.

    Only considers messages from the CURRENT turn (after the last HumanMessage)
    to properly isolate context when conversation memory is active.
    """
    messages = list(state["messages"])

    # Find the last human message to scope to current turn
    last_human_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i

    current_turn = messages[last_human_idx:]

    tool_outputs = []
    tools_used = []
    sources = []

    for msg in current_turn:
        if not isinstance(msg, ToolMessage):
            continue

        tool_outputs.append(msg.content)
        tool_name = msg.name if hasattr(msg, "name") else "unknown"
        if tool_name not in tools_used:
            tools_used.append(tool_name)

        # Extract badge names from tool output
        badge_names = re.findall(r"Badge:\s*(.+?)(?:\n|$)", msg.content)
        badge_name = badge_names[0].strip() if badge_names else None
        # Clean parenthetical key suffix if present
        if badge_name and "(" in badge_name:
            badge_name = badge_name.split("(")[0].strip()

        # Extract relevance scores from semantic search output
        scores = re.findall(r"relevance:\s*([\d.]+)", msg.content)
        match_score = float(scores[0]) if scores else None

        snippet = msg.content[:200]
        if len(msg.content) > 200:
            snippet += "..."

        sources.append(
            RetrievalSource(
                tool_name=tool_name,
                badge_name=badge_name,
                match_score=match_score,
                snippet=snippet,
            )
        )

    return tool_outputs, tools_used, sources


def _compute_grounded_confidence(
    tool_outputs: list[str], tools_used: list[str]
) -> tuple[float, str]:
    """Compute confidence from retrieval signals — NOT from LLM hallucination.

    Scoring logic:
    - No tool data → 0.2
    - Tools ran but no match → 0.3
    - Semantic match only → 0.65
    - Exact badge match → 0.85
    - Both tools corroborate → +0.1 bonus
    - Multiple sources → +0.05 bonus
    """
    if not tool_outputs:
        return 0.2, "No tool data retrieved; low confidence."

    has_exact = "get_alternative_docs" in tools_used
    has_semantic = "semantic_docs_search" in tools_used
    has_badge_data = any("Badge:" in out for out in tool_outputs)
    has_no_match = any(
        "No badge found" in out or "No matching" in out for out in tool_outputs
    )

    if has_no_match and not has_badge_data:
        return 0.3, "Query did not match any known specifications."

    confidence = 0.5
    rationale_parts = []

    if has_exact and has_badge_data:
        confidence = 0.85
        rationale_parts.append("Exact badge match via deterministic lookup")

    if has_semantic and has_exact:
        confidence = min(confidence + 0.1, 1.0)
        rationale_parts.append("corroborated by semantic search")
    elif has_semantic and not has_exact:
        confidence = 0.65
        rationale_parts.append("Semantic match only (no exact badge lookup)")

    if len(tool_outputs) > 2:
        confidence = min(confidence + 0.05, 1.0)
        rationale_parts.append(f"{len(tool_outputs)} sources consulted")

    rationale = (
        "; ".join(rationale_parts)
        if rationale_parts
        else "Based on retrieved documentation."
    )
    return round(confidence, 2), rationale


# ---------------------------------------------------------------------------
# 5. Context-Isolated Synthesis Node
# ---------------------------------------------------------------------------

def synthesize_output(state: AgentState) -> dict:
    """Synthesizes the final response using ONLY grounded tool outputs.

    This node enforces hallucination mitigation by:
    1. Extracting only ToolMessage content (never raw LLM speculation)
    2. Computing confidence from retrieval signals
    3. Overriding any LLM-generated confidence with the grounded value
    """
    tool_outputs, tools_used, sources = _extract_retrieval_context(state)

    # Find the user's query (last HumanMessage)
    user_query = ""
    for msg in reversed(list(state["messages"])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Compute grounded confidence
    confidence, rationale = _compute_grounded_confidence(tool_outputs, tools_used)

    # Collect all referenced badges
    all_badges = list({s.badge_name for s in sources if s.badge_name})

    # Build context-isolated prompt — ONLY tool outputs
    grounded_context = (
        "\n---\n".join(tool_outputs) if tool_outputs else "No documentation was retrieved."
    )

    formatting_instruction = (
        "You are a response formatter for a documentation intelligence system.\n"
        "Do NOT call any tools. Do NOT fabricate data.\n"
        "Use ONLY the retrieved documentation below to populate the response.\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"--- RETRIEVED DOCUMENTATION ---\n{grounded_context}\n\n"
        "Instructions:\n"
        "- direct_answer: Provide a clear, conversational answer citing specific documents and options.\n"
        "- relevant_sections: List the specific badge names or specification keys found.\n"
        "- suggested_followups: Suggest 2-3 related questions the user might ask next.\n"
        "- badges_referenced: List badge names consulted.\n"
        f"- confidence_score: Use {confidence}\n"
        f"- confidence_rationale: Use '{rationale}'\n"
    )

    try:
        structured_res = model_with_structured_output.invoke(
            [HumanMessage(content=formatting_instruction)]
        )

        # Override LLM confidence with grounded computation
        structured_res.confidence_score = confidence
        structured_res.confidence_rationale = rationale
        structured_res.retrieval_sources = sources
        structured_res.badges_referenced = all_badges

        logger.info(
            f"Synthesis complete: confidence={confidence}, "
            f"sources={len(sources)}, badges={all_badges}"
        )

    except Exception as e:
        logger.error(f"Structured output synthesis failed: {e}")
        # Fallback: return a valid response even if structured output fails
        structured_res = AgentDocResponse(
            direct_answer=(
                f"I retrieved documentation but encountered a formatting error: {str(e)}\n\n"
                f"Raw retrieval data:\n{grounded_context[:500]}"
            ),
            confidence_score=confidence,
            confidence_rationale=rationale,
            retrieval_sources=sources,
            badges_referenced=all_badges,
        )

    return {"structured_response": structured_res}


# ---------------------------------------------------------------------------
# 6. Assemble the StateGraph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("synthesize", synthesize_output)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "synthesize": "synthesize",
    },
)
workflow.add_edge("tools", "agent")
workflow.add_edge("synthesize", END)

# Compile with MemorySaver for conversation persistence across queries
memory = MemorySaver()
doc_agent = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# 7. Public Entry Point
# ---------------------------------------------------------------------------

def run_doc_agent(query: str, thread_id: str = "default") -> AgentDocResponse:
    """Run a query through the LangGraph pipeline with conversation memory.

    Args:
        query: The user's natural language query.
        thread_id: Session identifier for conversation memory continuity.

    Returns:
        Structured AgentDocResponse with grounded confidence and provenance.
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "structured_response": None,
        "iteration_count": 0,
        "retrieval_context": [],
        "tools_invoked": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Agent invoked: thread={thread_id}, query={query[:80]}")

    result = doc_agent.invoke(initial_state, config=config)
    return result["structured_response"]