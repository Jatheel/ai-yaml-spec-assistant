import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.alternative_docs_tool import get_alternative_docs
from tools.semantic_search_tool import semantic_docs_search
from agent.schemas import AgentDocResponse

load_dotenv()


class AgentState(TypedDict):
    """Internal graph state holding conversation history and output model."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    structured_response: AgentDocResponse | None


# 1. Initialize Tools & LLM
tools = [get_alternative_docs, semantic_docs_search]
tool_node = ToolNode(tools)

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)
model_with_tools = llm.bind_tools(tools)
model_with_structured_output = llm.with_structured_output(AgentDocResponse)

SYSTEM_PROMPT = (
    "You are a specialized documentation intelligence assistant.\n"
    "Your duty is to query and examine internal YAML specifications using the available "
    "retrieval tools before answering.\n"
    "Always rely strictly on factual data retrieved from the documentation tools. "
    "Never fabricate or guess specifications."
)


# 2. Define Graph Nodes
def call_model(state: AgentState):
    """Processes message history and determines if tool invocation is necessary."""
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Evaluates if the model generated tool calls or is ready to synthesize output."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesize"


def synthesize_output(state: AgentState):
    """Extracts final grounded facts and enforces structured Pydantic schema."""
    messages = state["messages"]
    formatting_instruction = (
        "Based on the conversation and tool outputs above, compile the definitive "
        "structured response conforming to the AgentDocResponse schema."
    )
    structured_res = model_with_structured_output.invoke(
        list(messages) + [HumanMessage(content=formatting_instruction)]
    )
    return {"structured_response": structured_res}


# 3. Assemble the StateGraph
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
        "synthesize": "synthesize"
    }
)
workflow.add_edge("tools", "agent")
workflow.add_edge("synthesize", END)

doc_agent = workflow.compile()


def run_doc_agent(query: str) -> AgentDocResponse:
    """Entrypoint function to run a query through the LangGraph documentation pipeline."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "structured_response": None
    }
    result = doc_agent.invoke(initial_state)
    return result["structured_response"]