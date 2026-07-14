"""Validated schemas for explainable opportunity scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, root_validator


class ScoreComponent(BaseModel):
    """One normalized score with a human-readable explanation and inputs."""

    score: float = Field(ge=0.0, le=100.0)
    reason: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class ProblemScoreWeights(BaseModel):
    """Weights for the four Problem Score components."""

    pain_severity: float = Field(0.35, ge=0.0)
    problem_frequency: float = Field(0.25, ge=0.0)
    willingness_to_pay: float = Field(0.20, ge=0.0)
    evidence_quality: float = Field(0.20, ge=0.0)

    @root_validator(allow_reuse=True)
    def total_must_equal_one(cls, values: dict[str, float]) -> dict[str, float]:
        """Reject partial or inflated weight sets."""

        if abs(sum(values.values()) - 1.0) > 1e-8:
            raise ValueError("Problem Score weights must total 1.0.")
        return values


class OpportunityScoreWeights(BaseModel):
    """Weights for the initial Opportunity Score."""

    pain_severity: float = Field(0.25, ge=0.0)
    problem_frequency: float = Field(0.15, ge=0.0)
    willingness_to_pay: float = Field(0.15, ge=0.0)
    evidence_quality: float = Field(0.10, ge=0.0)
    whitespace: float = Field(0.15, ge=0.0)
    build_feasibility: float = Field(0.10, ge=0.0)
    market_accessibility: float = Field(0.10, ge=0.0)

    @root_validator(allow_reuse=True)
    def total_must_equal_one(cls, values: dict[str, float]) -> dict[str, float]:
        """Reject partial or inflated weight sets."""

        if abs(sum(values.values()) - 1.0) > 1e-8:
            raise ValueError("Opportunity Score weights must total 1.0.")
        return values


class ProblemScoreBreakdown(BaseModel):
    """Complete Problem Score output."""

    pain_severity: ScoreComponent
    problem_frequency: ScoreComponent
    willingness_to_pay: ScoreComponent
    evidence_quality: ScoreComponent
    problem_score: ScoreComponent


class WhiteSpaceScoreBreakdown(BaseModel):
    """Complete five-component White-Space Score output."""

    unmet_customer_need: ScoreComponent
    differentiation_potential: ScoreComponent
    competitor_weakness: ScoreComponent
    niche_specificity: ScoreComponent
    low_direct_competitor_density: ScoreComponent
    whitespace_score: ScoreComponent


class OpportunityScoringResult(BaseModel):
    """Persistable output from initial opportunity scoring."""

    pain_severity_score: float = Field(ge=0.0, le=100.0)
    problem_frequency_score: float = Field(ge=0.0, le=100.0)
    willingness_to_pay_score: float = Field(ge=0.0, le=100.0)
    evidence_quality_score: float = Field(ge=0.0, le=100.0)
    whitespace_score: float = Field(ge=0.0, le=100.0)
    build_feasibility_score: float = Field(ge=0.0, le=100.0)
    market_accessibility_score: float = Field(ge=0.0, le=100.0)
    opportunity_score: float = Field(ge=0.0, le=100.0)
    confidence_score: float = Field(ge=0.0, le=100.0)
    explanation_json: dict[str, Any]
