import json
import logging
from datetime import datetime
from typing import Callable

import anthropic
from crewai import Agent
from pydantic import BaseModel, Field

from agents.ceo_orchestrator import FounderRequest
from context.memory import MemoryContent

logger = logging.getLogger(__name__)

_LLM_MODEL = "claude-opus-4-6"
_MAX_CHECKPOINT_RETRIES = 3

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AcceptanceCriteria(BaseModel):
    """A single acceptance criterion in Given/When/Then format."""

    given: str
    when: str
    then: str


class UserStory(BaseModel):
    """A user story with acceptance criteria."""

    id: str  # format: US-001, US-002, etc.
    title: str
    as_a: str
    i_want: str
    so_that: str
    acceptance_criteria: list[AcceptanceCriteria]
    story_points: int | None = None  # estimated by planner later


class PRDOutput(BaseModel):
    """Structured Product Requirements Document produced by the ProductThinker."""

    feature_name: str
    problem_statement: str
    proposed_solution: str
    out_of_scope: list[str]
    user_stories: list[UserStory]
    open_questions: list[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ThinkerInput(BaseModel):
    """Input consumed by the ProductThinker agent."""

    request: FounderRequest
    memory: MemoryContent


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConflictError(Exception):
    """Raised when a PRD proposes features that conflict with product memory."""

    def __init__(self, message: str, conflicts: list[str] | None = None) -> None:
        super().__init__(message)
        self.conflicts: list[str] = conflicts or []


class CheckpointError(Exception):
    """Raised when the human checkpoint fails after the maximum number of retries."""

    def __init__(self, message: str, prd: PRDOutput | None = None) -> None:
        super().__init__(message)
        self.prd = prd


# ---------------------------------------------------------------------------
# ProductThinker
# ---------------------------------------------------------------------------


class ProductThinker:
    """First agent in the pipeline: transforms a FounderRequest into a PRD.

    Reads the product memory before any reasoning to avoid proposing features
    that already exist or were discarded. Produces a structured PRDOutput with
    user stories and acceptance criteria, then waits for human approval via
    a checkpoint before passing the PRD downstream.
    """

    def __init__(self, llm_client: anthropic.Anthropic) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input: ThinkerInput) -> PRDOutput:  # noqa: A002
        """Runs the LLM reasoning loop and returns an approved PRDOutput.

        Raises:
            ConflictError: if the proposed feature conflicts with memory.
        """
        prompt = self._build_prompt(input)
        prd = self._call_llm(prompt)

        conflicts = self._check_conflicts(prd, input.memory)
        if conflicts:
            raise ConflictError(
                f"PRD conflicts with product memory: {'; '.join(conflicts)}",
                conflicts=conflicts,
            )

        logger.info("ProductThinker produced PRD: %s", prd.feature_name)
        return prd

    def checkpoint(
        self,
        prd: PRDOutput,
        notify_fn: Callable[[str], None],
        poll_fn: Callable[[], str | None],
        original_input: ThinkerInput,
    ) -> PRDOutput:
        """Sends the PRD to Slack and waits for founder approval.

        Args:
            prd: The initial PRD to submit for review.
            notify_fn: Callable that posts a message to Slack.
            poll_fn: Callable that blocks until the founder replies, returning
                     the raw reply text or None on timeout.
            original_input: The original ThinkerInput used to re-run on rejection.

        Returns:
            The approved PRDOutput (may differ from input if re-runs occurred).

        Raises:
            CheckpointError: if the PRD is rejected more than _MAX_CHECKPOINT_RETRIES times.
        """
        current_prd = prd

        for attempt in range(1, _MAX_CHECKPOINT_RETRIES + 1):
            summary = self._format_prd_summary(current_prd)
            notify_fn(
                f"📋 PRD ready for review (attempt {attempt}/{_MAX_CHECKPOINT_RETRIES}):\n\n"
                f"{summary}\n\n"
                "Reply *approve* or *reject: <your feedback>*"
            )

            response = poll_fn()
            if response is None:
                raise CheckpointError(
                    "PRD checkpoint timed out — no response from founder.",
                    prd=current_prd,
                )

            normalized = response.strip().lower()

            if normalized == "approve":
                logger.info("PRD approved by founder on attempt %d.", attempt)
                return current_prd

            if normalized.startswith("reject:"):
                feedback = response[len("reject:"):].strip()
                logger.info(
                    "PRD rejected on attempt %d. Feedback: %s", attempt, feedback
                )
                if attempt < _MAX_CHECKPOINT_RETRIES:
                    prompt = self._build_prompt(original_input, feedback=feedback)
                    current_prd = self._call_llm(prompt)
                continue

            # Unrecognised reply — ask again without consuming an attempt.
            notify_fn(
                "🤔 Please reply with *approve* or *reject: <your feedback>* to continue."
            )

        raise CheckpointError(
            f"PRD rejected {_MAX_CHECKPOINT_RETRIES} consecutive times without approval.",
            prd=current_prd,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self, input: ThinkerInput, feedback: str | None = None  # noqa: A002
    ) -> str:
        """Constructs the LLM prompt from the ThinkerInput and optional feedback."""
        memory = input.memory
        req = input.request

        implemented_text = (
            "\n".join(
                f"  - {f.name}: {f.description}"
                for f in memory.implemented_features
            )
            or "  None yet."
        )
        discarded_text = (
            "\n".join(
                f"  - {f.name}: {f.reason}"
                for f in memory.discarded_features
            )
            or "  None yet."
        )
        decisions_text = (
            "\n".join(
                f"  - {d.decision}: {d.rationale}"
                for d in memory.architectural_decisions
            )
            or "  None yet."
        )
        patterns_text = (
            "\n".join(
                f"  - {p.name}: {p.description}"
                for p in memory.naming_patterns
            )
            or "  None yet."
        )

        feedback_section = (
            f"\n## Previous Rejection Feedback\n{feedback}\n"
            "Revise your output to address this feedback.\n"
            if feedback
            else ""
        )

        schema_example = json.dumps(
            {
                "feature_name": "string",
                "problem_statement": "string",
                "proposed_solution": "string",
                "out_of_scope": ["string"],
                "user_stories": [
                    {
                        "id": "US-001",
                        "title": "string",
                        "as_a": "string",
                        "i_want": "string",
                        "so_that": "string",
                        "acceptance_criteria": [
                            {"given": "string", "when": "string", "then": "string"}
                        ],
                        "story_points": None,
                    }
                ],
                "open_questions": ["string"],
                "created_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        )

        return f"""You are a senior product manager with deep knowledge of the existing product.
Your task is to produce a structured PRD in response to a founder's request.

## Product Memory

### Implemented Features (DO NOT re-propose — they already exist):
{implemented_text}

### Discarded Features (DO NOT re-propose — they were deliberately rejected):
{discarded_text}

### Architectural Decisions:
{decisions_text}

### Naming & UX Patterns:
{patterns_text}

## Founder's Request
Problem: {req.problem}
Tech Stack: {req.stack or "Not specified"}
Repository: {req.repo or "Not specified"}
Priority: {req.priority}
{feedback_section}
## Instructions
1. Read the product memory and identify any conflicts with existing or discarded features.
2. Identify the REAL underlying problem — it may differ from what was literally asked.
3. Define MINIMUM VIABLE SCOPE. Avoid over-engineering or scope creep.
4. Write user stories in "As a / I want / So that" format.
5. Write acceptance criteria in "Given / When / Then" format.
6. Explicitly list what is OUT OF SCOPE for this feature.
7. Flag any open questions that require founder input before development starts.
8. NEVER propose features already in Implemented Features or Discarded Features.
9. Include at least 2 user stories, each with at least 1 acceptance criterion.

## Output Format
Return ONLY a valid JSON object. No markdown, no explanation, just the JSON.
Schema:
{schema_example}
"""

    def _call_llm(self, prompt: str) -> PRDOutput:
        """Calls the Anthropic API and parses the response into a PRDOutput."""
        message = self._llm.messages.create(
            model=_LLM_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> PRDOutput:
        """Extracts and deserialises the JSON PRDOutput from a raw LLM response."""
        text = raw.strip()

        # Strip markdown code fences if the LLM wrapped the JSON.
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(text)
        return PRDOutput.model_validate(data)

    def _check_conflicts(
        self, prd: PRDOutput, memory: MemoryContent
    ) -> list[str]:
        """Returns a list of conflict descriptions between the PRD and memory."""
        conflicts: list[str] = []
        needle = prd.feature_name.lower()

        for feature in memory.implemented_features:
            if feature.name.lower() in needle or needle in feature.name.lower():
                conflicts.append(
                    f"'{prd.feature_name}' conflicts with already-implemented "
                    f"feature '{feature.name}'."
                )

        for feature in memory.discarded_features:
            if feature.name.lower() in needle or needle in feature.name.lower():
                conflicts.append(
                    f"'{prd.feature_name}' was previously discarded: {feature.reason}"
                )

        return conflicts

    @staticmethod
    def _format_prd_summary(prd: PRDOutput) -> str:
        """Formats a PRDOutput as a human-readable Slack message."""
        stories = "\n".join(
            f"  • *{s.id}* — {s.title}\n"
            f"    As a {s.as_a}, I want {s.i_want}, so that {s.so_that}"
            for s in prd.user_stories
        )
        out_of_scope = "\n".join(f"  - {item}" for item in prd.out_of_scope)
        questions = "\n".join(f"  - {q}" for q in prd.open_questions)

        return (
            f"*Feature:* {prd.feature_name}\n\n"
            f"*Problem:* {prd.problem_statement}\n\n"
            f"*Solution:* {prd.proposed_solution}\n\n"
            f"*User Stories:*\n{stories}\n\n"
            f"*Out of Scope:*\n{out_of_scope or '  None listed.'}\n\n"
            f"*Open Questions:*\n{questions or '  None.'}"
        )


# ---------------------------------------------------------------------------
# CrewAI wrapper
# ---------------------------------------------------------------------------


class ProductThinkerAgent:
    """CrewAI Agent wrapper for the ProductThinker.

    The reasoning logic lives in ProductThinker. This class exposes a build()
    method that returns a configured crewai.Agent for use inside ProdutoCrew.
    """

    def build(self) -> Agent:
        """Returns a configured CrewAI Agent for the product-thinking role."""
        return Agent(
            role="Product Thinker",
            goal=(
                "Transform founder requests into structured PRDs that respect "
                "product memory and avoid re-proposing existing or discarded features."
            ),
            backstory=(
                "You are a senior product manager who deeply understands the existing "
                "product. You always read the product memory before reasoning. You never "
                "propose features that already exist or were explicitly discarded. You "
                "identify the real problem behind every request and define minimum "
                "viable scope."
            ),
            llm=_LLM_MODEL,
            memory=True,
            max_iter=3,
            verbose=True,
        )
