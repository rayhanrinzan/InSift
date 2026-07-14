"""Pydantic schemas for competitor research."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Normalized result returned by any search provider."""

    title: str
    url: str
    snippet: str = ""
    content: Optional[str] = None
    score: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompetitorResearchContext(BaseModel):
    """Opportunity fields needed to classify search results."""

    title: str
    problem_summary: str
    target_customer: Optional[str] = None
    current_workaround: Optional[str] = None
    proposed_solution: Optional[str] = None


class CompetitorClassification(BaseModel):
    """Structured classification for a potential competitor result."""

    company_name: Optional[str] = None
    product_name: Optional[str] = None
    relationship_type: Literal["direct", "adjacent", "substitute", "irrelevant"]
    target_customer: Optional[str] = None
    problem_solved: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    pricing_position: Optional[str] = None
    similarity_score: float = Field(0.0, ge=0.0, le=1.0)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    possible_gap: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str
