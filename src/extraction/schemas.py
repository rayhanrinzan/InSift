"""Pydantic schemas for structured extraction outputs."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


PainType = Literal[
    "time",
    "labor",
    "cost",
    "lost_revenue",
    "risk",
    "compliance",
    "coordination",
    "data_entry",
    "poor_user_experience",
    "lack_of_visibility",
    "integration",
    "repetitive_work",
]


class ExtractedProblem(BaseModel):
    """Structured problem extracted from source text."""

    contains_real_problem: bool
    problem_statement: Optional[str] = None
    affected_user: Optional[str] = None
    current_workaround: Optional[str] = None
    pain_types: list[PainType] = Field(default_factory=list)
    severity_score: float = Field(0.0, ge=0.0, le=1.0)
    frequency_signal: float = Field(0.0, ge=0.0, le=1.0)
    willingness_to_pay_score: float = Field(0.0, ge=0.0, le=1.0)
    evidence_quote: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def has_usable_problem(self) -> bool:
        """Return whether the extraction contains a grounded problem statement."""

        return bool(self.contains_real_problem and self.problem_statement)
