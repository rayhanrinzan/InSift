"""Evidence-grounded opportunity synthesis for persisted problem clusters."""

from __future__ import annotations

import json
import html
import re
from collections import Counter
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


class ResilientOpportunitySynthesizer:
    """Use OpenAI synthesis when possible and conservative local synthesis otherwise."""

    def __init__(
        self,
        primary: OpportunitySynthesisProvider,
        fallback: OpportunitySynthesisProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_available = True

    def synthesize(
        self,
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
    ) -> OpportunityDraft:
        if not self.primary_available:
            return self.fallback.synthesize(cluster, evidence_items)
        try:
            return self.primary.synthesize(cluster, evidence_items)
        except OpportunitySynthesisError:
            self.primary_available = False
            return self.fallback.synthesize(cluster, evidence_items)


class DeterministicOpportunitySynthesizer:
    """Conservative local synthesis that requires repeated specific language."""

    GENERIC_TERMS = {
        "customer",
        "customers",
        "every",
        "frustrating",
        "hours",
        "manual",
        "manually",
        "operations",
        "problem",
        "process",
        "repetitive",
        "spreadsheet",
        "spreadsheets",
        "still",
        "takes",
        "team",
        "teams",
        "using",
        "workaround",
    }

    def synthesize(
        self,
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
    ) -> OpportunityDraft:
        shared_terms = self._shared_specific_terms(evidence_items)
        workflow_topic = self._shared_workflow_topic(evidence_items)
        supported = len(evidence_items) >= 2 and bool(
            workflow_topic or len(shared_terms) >= 2
        )
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
        workaround = self._clean_workaround(workaround)
        problem = self._problem_summary(
            cluster,
            evidence_items,
            target=target,
            workflow_topic=workflow_topic,
        )
        display_workflow = self._display_workflow(workflow_topic or "")
        title = (
            f"{target} struggle with {display_workflow}"
            if workflow_topic
            else cluster.title
        )[:120]
        product_focus = display_workflow or "the documented workflow"
        return OpportunityDraft(
            supported=supported,
            title=title,
            problem_summary=problem,
            target_customer=target,
            current_workaround=workaround,
            proposed_solution=(
                f"A focused {product_focus} workspace for {target} that replaces "
                f"{workaround} with one shared queue, explicit ownership, automatic "
                "routine handoffs, and exception alerts. Start with one integration "
                "used by pilot teams and measure handling time and missed work."
            )[:1_500],
            reasoning=(
                f"The cluster contains {len(evidence_items)} independently sourced "
                "accepted records. Repeated specific terms: "
                f"{', '.join(shared_terms[:8]) or 'none'}."
            ),
            confidence=(
                min(0.88, 0.55 + len(shared_terms) * 0.04)
                if supported
                else 0.35
            ),
        )

    @staticmethod
    def _shared_workflow_topic(evidence_items: list[EvidenceItem]) -> str | None:
        topics = [
            str((item.metadata_json or {}).get("scout_workflow_topic") or "").strip()
            for item in evidence_items
        ]
        topics = [topic for topic in topics if topic]
        if not topics:
            return None
        topic, frequency = Counter(topics).most_common(1)[0]
        return topic if frequency >= 2 else None

    @staticmethod
    def _clean_workaround(value: str) -> str:
        cleaned = re.sub(
            r"^Uses\s+|\s+according to the source text$",
            "",
            value.strip(),
            flags=re.IGNORECASE,
        )
        if cleaned.lower() == "excel":
            return "Excel spreadsheets"
        if cleaned.lower() == "spreadsheet":
            return "spreadsheets"
        return cleaned or "manual coordination"

    @staticmethod
    def _display_workflow(value: str) -> str:
        return (
            value.replace(" customer emails", " and customer updates")
            .replace(" follow up", " follow-up")
            .replace(" follow-up follow-up", " follow-up")
        )

    @staticmethod
    def _problem_summary(
        cluster: OpportunityCluster,
        evidence_items: list[EvidenceItem],
        *,
        target: str,
        workflow_topic: str | None,
    ) -> str:
        if not workflow_topic:
            return cluster.problem_summary
        excerpts: list[str] = []
        for item in evidence_items:
            text = DeterministicOpportunitySynthesizer._evidence_excerpt(item)
            if text and text not in excerpts:
                excerpts.append(text)
            if len(excerpts) >= 2:
                break
        evidence_summary = " Another source reports: ".join(excerpts)
        display_workflow = DeterministicOpportunitySynthesizer._display_workflow(
            workflow_topic
        )
        return (
            f"{target} report recurring failures in {display_workflow}. "
            f"{evidence_summary}"
        )[:1_500]

    @staticmethod
    def _evidence_excerpt(item: EvidenceItem) -> str:
        text = " ".join((item.problem_statement or "").split())
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        source_title = " ".join(html.unescape(item.title or "").split()).strip(" .")
        for _ in range(2):
            if source_title and text.lower().startswith(source_title.lower()):
                text = text[len(source_title) :].lstrip(" .:-")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        selected: list[str] = []
        for sentence in sentences:
            candidate = " ".join((*selected, sentence))
            if selected and len(candidate) > 260:
                break
            selected.append(sentence)
            if len(candidate) >= 140:
                break
        excerpt = " ".join(selected) or text
        if len(excerpt) > 280:
            excerpt = f"{excerpt[:277].rsplit(' ', 1)[0]}..."
        return excerpt

    @classmethod
    def _shared_specific_terms(
        cls,
        evidence_items: list[EvidenceItem],
    ) -> list[str]:
        document_frequency: Counter[str] = Counter()
        for item in evidence_items:
            text = " ".join(
                filter(None, (item.problem_statement, item.affected_user))
            ).lower()
            tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", text)
                if len(token) >= 4 and token not in cls.GENERIC_TERMS
            }
            document_frequency.update(tokens)
        return sorted(
            token for token, frequency in document_frequency.items() if frequency >= 2
        )


def build_opportunity_synthesizer(
    settings: Settings,
) -> OpportunitySynthesisProvider:
    """Build evidence synthesis with a quota-independent local fallback."""

    local = DeterministicOpportunitySynthesizer()
    if (
        settings.demo_mode
        or (settings.llm_provider or "").lower() != "openai"
        or not settings.llm_api_key
    ):
        return local
    primary = OpenAIOpportunitySynthesizer(
        OpenAIClient(
            settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            base_url=settings.openai_base_url,
        )
    )
    return ResilientOpportunitySynthesizer(primary, local)
