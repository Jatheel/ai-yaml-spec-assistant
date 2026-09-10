"""Structured response schemas for the documentation intelligence agent."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievalSource(BaseModel):
    """Tracks which tool provided what data, enabling confidence grounding and provenance."""

    tool_name: str = Field(
        description="Name of the retrieval tool that produced this data."
    )
    badge_name: Optional[str] = Field(
        default=None,
        description="Certification badge referenced, if applicable."
    )
    match_score: Optional[float] = Field(
        default=None,
        description="Relevance or confidence score from the retrieval tool."
    )
    snippet: str = Field(
        default="",
        description="Brief excerpt of the retrieved content for provenance tracking."
    )


class AgentDocResponse(BaseModel):
    """Production-grade structured response schema returned by the documentation assistant."""

    direct_answer: str = Field(
        description="Direct and concise answer to the user query based on the documentation."
    )
    relevant_sections: List[str] = Field(
        default_factory=list,
        description="List of relevant section titles, specification keys, or document blocks used."
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0 reflecting factual grounding in the documents."
    )
    confidence_rationale: str = Field(
        default="",
        description="Brief explanation of why this confidence level was assigned."
    )
    retrieval_sources: List[RetrievalSource] = Field(
        default_factory=list,
        description="Provenance chain tracking which tools and sources provided the data."
    )
    badges_referenced: List[str] = Field(
        default_factory=list,
        description="List of certification badge names consulted during retrieval."
    )
    suggested_followups: Optional[List[str]] = Field(
        default_factory=list,
        description="Suggested follow-up questions or related topics for the user."
    )