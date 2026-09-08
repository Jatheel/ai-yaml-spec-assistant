from typing import List, Optional
from pydantic import BaseModel, Field


class AgentDocResponse(BaseModel):
    """Structured response schema returned by the documentation assistant agent."""

    direct_answer: str = Field(
        description="Direct and concise answer to the user query based on the documentation."
    )
    relevant_sections: List[str] = Field(
        default_factory=list,
        description="List of relevant section titles, specification keys, or document blocks used."
    )
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 reflecting factual grounding in the documents."
    )
    suggested_followups: Optional[List[str]] = Field(
        default_factory=list,
        description="Suggested follow-up questions or related topics for the user."
    )