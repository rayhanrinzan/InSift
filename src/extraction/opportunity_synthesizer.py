"""Evidence-grounded opportunity synthesis for persisted problem clusters."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from src.config import Settings
from src.database.models import EvidenceItem, OpportunityCluster
from src.providers.openai import OpenAIClient, OpenAIProviderError


class OpportunitySynthesisError(RuntimeError):
    """Raised when a grounded opportunity summary cannot be produced."""


class OpportunityDraft(BaseModel):
    """Problem and product direction synthesized from linked evidence only."""

    supported: bool
    title: str = Field(min_length=5, max_length=120)
    problem_summary: str = Field(min_length=20, max_length=1_500)
    target_customer: str = Field(min_length=3, max_length=255)
    current_workaround: str = Field(min_length=3, max_length=1_000)
    proposed_solution: str = Field(min_length=20, max_length=1_500)
    reasoning: str = Field(min_length=10, max_length=1_000)
    confidence: float = Field(ge=0.0, le=1.0)


class OpportunitySynthesisProvider(Protocol):
    """Synthesize one coherent opportunity from a persisted evidence cluster."""

    def synthesize(
        self,
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
    ) -> OpportunityDraft:
        """Return a grounded problem and product direction."""


OPPORTUNITY_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "supported": {"type": "boolean"},
        "title": {"type": "string"},
        "problem_summary": {"type": "string"},
        "target_customer": {"type": "string"},
        "current_workaround": {"type": "string"},
        "proposed_solution": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "supported",
        "title",
        "problem_summary",
        "target_customer",
        "current_workaround",
        "proposed_solution",
        "reasoning",
        "confidence",
    ],
}

OPPORTUNITY_SYNTHESIS_PROMPT = """
Synthesize one startup opportunity from the supplied accepted evidence records.
Use only facts supported by those records. The problem summary must describe the
recurring operational problem shared by the sources, not the search query that found
them. Name a specific target customer and current workaround only when supported.
The proposed solution is a narrow MVP hypothesis that directly replaces or improves
the documented workaround; do not claim proven demand, market size, novelty, or
competitor absence. Keep the title concrete and concise. Explain which repeated
evidence supports the synthesis. Reduce confidence when sources are vague or weakly
aligned. Set supported to false when the records are generic, unrelated, concern
different underlying workflows, or do not contain repeated first-hand problem
evidence. Never force an opportunity from weak search results.
""".strip()


class OpenAIOpportunitySynthesizer:
    """Use OpenAI structured output to synthesize an evidence-backed opportunity."""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def synthesize(
        self,
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
    ) -> OpportunityDraft:
        payload = {
            "current_cluster": {
                "title": cluster.title,
                "problem_summary": cluster.problem_summary,
            },
            "evidence": [
                {
                    "source_url": item.source_url,
                    "source_site": item.community or item.platform,
                    "scout_segment": (item.metadata_json or {}).get(
                        "scout_segment_label"
                    ),
                    "problem_statement": (item.problem_statement or "")[:1_500],
                    "affected_user": (item.affected_user or "")[:255],
                    "current_workaround": (item.current_workaround or "")[:1_000],
                    "pain_types": item.pain_types or [],
                    "evidence_quote": str(
                        (item.metadata_json or {}).get("evidence_quote") or ""
                    )[:1_500],
                }
                for item in evidence_items[:12]
            ],
        }
        try:
            response = self.client.structured_response(
                schema_name="opportunity_draft",
                schema=OPPORTUNITY_DRAFT_SCHEMA,
                instructions=OPPORTUNITY_SYNTHESIS_PROMPT,
                input_text=json.dumps(payload, ensure_ascii=True),
            )
            return OpportunityDraft.parse_obj(response)
        except ValidationError as exc:
            raise OpportunitySynthesisError(
                "OpenAI returned an invalid opportunity synthesis."
            ) from exc
        except OpenAIProviderError as exc:
            raise OpportunitySynthesisError(str(exc)) from exc


class DeterministicOpportunitySynthesizer:
    """Predictable evidence-grounded synthesis used only by automated tests."""

    def synthesize(
        self,
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
    ) -> OpportunityDraft:
        target = next(
            (item.affected_user for item in evidence_items if item.affected_user),
            None,
        ) or cluster.target_customer or "operations teams"
        workaround = next(
            (
                item.current_workaround
                for item in evidence_items
                if item.current_workaround
            ),
            None,
        ) or cluster.current_workaround or "manual coordination"
        problem = cluster.problem_summary
        return OpportunityDraft(
            supported=True,
            title=cluster.title,
            problem_summary=problem,
            target_customer=target,
            current_workaround=workaround,
            proposed_solution=(
                f"A focused workflow for {target} that replaces {workaround} and "
                f"directly addresses the documented problem: {problem}"
            )[:1_500],
            reasoning=(
                f"The cluster contains {len(evidence_items)} independently sourced "
                "accepted evidence records describing the same problem."
            ),
            confidence=min(0.95, 0.55 + len(evidence_items) * 0.1),
        )


def build_opportunity_synthesizer(
    settings: Settings,
) -> OpportunitySynthesisProvider:
    """Build the live OpenAI synthesizer used by automatic problem scouting."""

    if settings.demo_mode or (settings.llm_provider or "").lower() != "openai":
        raise OpportunitySynthesisError(
            "Live opportunity synthesis requires Demo mode off and OpenAI configured."
        )
    if not settings.llm_api_key:
        raise OpportunitySynthesisError(
            "LLM_API_KEY is required for live opportunity synthesis."
        )
    return OpenAIOpportunitySynthesizer(
        OpenAIClient(
            settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            base_url=settings.openai_base_url,
        )
    )
