import json
import logging
import os
from datetime import datetime
from typing import Callable

import anthropic
from crewai import Agent
from pydantic import BaseModel, Field

from agents.product_thinker import AcceptanceCriteria
from agents.sprint_planner import SprintPlan, TechnicalTask
from context.memory import MemoryContent, NamingPattern
from context.rag_index import CodebaseRAG

logger = logging.getLogger(__name__)

_LLM_MODEL = "claude-opus-4-6"
_RAG_FETCH_TOP_K = 8
_MAX_CONTEXT_FILES = 6
_MIN_INSTRUCTIONS_LEN = 200
_PROMPT_REVIEW_ENV = "PROMPT_REVIEW_ENABLED"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CodeContext(BaseModel):
    """A single file from the RAG index, relevant to a task."""

    file_path: str
    language: str
    content: str
    relevance_score: float
    reason: str | None = None


class VibeCoderPrompt(BaseModel):
    """A complete, context-enriched prompt for the VibeCoder agent."""

    task_id: str
    task_title: str
    instructions: str
    context_files: list[CodeContext] = Field(default_factory=list)
    stack: str
    constraints: list[str] = Field(default_factory=list)
    expected_output: str
    acceptance_criteria: list[AcceptanceCriteria] = Field(default_factory=list)
    naming_patterns: list[NamingPattern] = Field(default_factory=list)
    estimated_files_to_modify: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PromptEngineerInput(BaseModel):
    """Input consumed by the PromptEngineer agent."""

    sprint_plan: SprintPlan
    memory: MemoryContent
    rag: CodebaseRAG
    stack: str

    model_config = {"arbitrary_types_allowed": True}


class PromptEngineerOutput(BaseModel):
    """Full output from one PromptEngineer run — one prompt per sprint task."""

    prompts: list[VibeCoderPrompt]
    total_context_files: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PromptEngineer
# ---------------------------------------------------------------------------


class PromptEngineer:
    """Third agent in the pipeline: converts sprint tasks into Vibe Coder prompts.

    Uses the RAG index to fetch relevant codebase files for each task and
    constructs precise, constraint-rich instructions so the VibeCoder produces
    consistent, well-integrated code.
    """

    def __init__(self, llm_client: anthropic.Anthropic) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input: PromptEngineerInput) -> PromptEngineerOutput:  # noqa: A002
        """Processes every task in the sprint plan and returns one prompt each."""
        prompts: list[VibeCoderPrompt] = []
        for task in input.sprint_plan.tasks:
            prompt = self._build_prompt_for_task(task, input)
            self._validate_prompt(prompt)
            prompts.append(prompt)

        all_paths = {f.file_path for p in prompts for f in p.context_files}
        logger.info(
            "PromptEngineer produced %d prompts referencing %d unique context files.",
            len(prompts),
            len(all_paths),
        )
        return PromptEngineerOutput(
            prompts=prompts,
            total_context_files=len(all_paths),
        )

    def optional_checkpoint(
        self,
        prompt: VibeCoderPrompt,
        notify_fn: Callable[[str], None],
        poll_fn: Callable[[], str | None],
        task: TechnicalTask,
        input: PromptEngineerInput,  # noqa: A002
    ) -> VibeCoderPrompt:
        """Optionally sends a prompt to Slack for human review before coding starts.

        Controlled by PROMPT_REVIEW_ENABLED env var (default: false).
        If disabled, returns the prompt unchanged immediately.
        If enabled, waits for "ok" or "refine: {feedback}" from the founder.
        """
        if os.getenv(_PROMPT_REVIEW_ENV, "false").lower() != "true":
            logger.info(
                "Prompt review disabled — skipping checkpoint for %s.", prompt.task_id
            )
            return prompt

        file_names = ", ".join(
            f.file_path.split("/")[-1] for f in prompt.context_files[:5]
        )
        top_constraints = "\n  ".join(prompt.constraints[:3])
        notify_fn(
            f"🔧 Prompt ready for task {prompt.task_id}: {prompt.task_title}\n"
            f"Context files ({len(prompt.context_files)}): {file_names}\n"
            f"Key constraints:\n  {top_constraints}\n"
            "Reply *ok* to proceed or *refine: <feedback>*"
        )

        response = poll_fn()
        if response is None:
            logger.warning("No response for prompt checkpoint %s — proceeding.", prompt.task_id)
            return prompt

        normalized = response.strip().lower()
        if normalized == "ok":
            return prompt

        if normalized.startswith("refine:"):
            feedback = response[len("refine:"):].strip()
            logger.info("Prompt refinement requested for %s: %s", prompt.task_id, feedback)
            context = self._fetch_context(task, input.rag)
            llm_prompt = self._build_llm_prompt(task, context, input.memory, feedback=feedback)
            return self._call_llm_and_parse(llm_prompt, task, context, input)

        logger.warning(
            "Unrecognised checkpoint response for %s: %s", prompt.task_id, response
        )
        return prompt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt_for_task(
        self, task: TechnicalTask, input: PromptEngineerInput  # noqa: A002
    ) -> VibeCoderPrompt:
        """Builds a single VibeCoderPrompt for one TechnicalTask."""
        context = self._fetch_context(task, input.rag)
        llm_prompt = self._build_llm_prompt(task, context, input.memory)
        return self._call_llm_and_parse(llm_prompt, task, context, input)

    def _call_llm_and_parse(
        self,
        llm_prompt: str,
        task: TechnicalTask,
        context: list[CodeContext],
        input: PromptEngineerInput,  # noqa: A002
    ) -> VibeCoderPrompt:
        """Calls the LLM and assembles the final VibeCoderPrompt."""
        message = self._llm.messages.create(
            model=_LLM_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": llm_prompt}],
        )
        raw = message.content[0].text
        prompt = self._parse_llm_response(raw, task, context)

        # Inject pre-populated fields that the LLM does not fill.
        prompt = prompt.model_copy(
            update={
                "context_files": context,
                "stack": input.stack,
                "constraints": self._build_constraints(task, input.memory),
                "acceptance_criteria": self._get_acceptance_criteria(
                    task, input.sprint_plan
                ),
                "naming_patterns": list(input.memory.naming_patterns),
            }
        )
        return prompt

    def _fetch_context(
        self, task: TechnicalTask, rag: CodebaseRAG
    ) -> list[CodeContext]:
        """Fetches and merges RAG results for a task, returning top-6 by score."""
        query = task.description

        general = rag.search(query, top_k=_RAG_FETCH_TOP_K)
        backend = rag.search_by_type(query, extensions=[".py"])
        frontend = rag.search_by_type(query, extensions=[".ts", ".tsx"])

        # Merge all results, keeping the highest score per unique file_path.
        best: dict[str, CodeContext] = {}
        all_results = general.files + backend.files + frontend.files
        for rf in all_results:
            existing = best.get(rf.file_path)
            if existing is None or rf.score > existing.relevance_score:
                best[rf.file_path] = CodeContext(
                    file_path=rf.file_path,
                    language=rf.language,
                    content=rf.content,
                    relevance_score=rf.score,
                    reason=self._infer_reason(rf.file_path, task),
                )

        if not best:
            logger.warning(
                "RAG returned 0 context files for task %s — index may be empty.",
                task.id,
            )

        sorted_files = sorted(
            best.values(), key=lambda c: c.relevance_score, reverse=True
        )
        return sorted_files[:_MAX_CONTEXT_FILES]

    def _build_constraints(
        self, task: TechnicalTask, memory: MemoryContent
    ) -> list[str]:
        """Assembles constraint strings from memory, task type, and universal rules."""
        constraints: list[str] = []

        # From naming patterns.
        for pattern in memory.naming_patterns:
            constraints.append(
                f"Follow naming pattern: {pattern.name} — {pattern.description}"
            )

        # From architectural decisions.
        for decision in memory.architectural_decisions:
            constraints.append(
                f"Respect architectural decision: {decision.decision}"
            )

        # Task-type inferred constraints.
        title_lower = task.title.lower()
        if "test" in title_lower:
            constraints.append(
                "Add unit tests with pytest, minimum coverage 75%"
            )
        if "api" in title_lower or "endpoint" in title_lower:
            constraints.append("Follow existing FastAPI route structure")
        if "frontend" in title_lower or "component" in title_lower:
            constraints.append("Follow existing React component patterns")
        if "migration" in title_lower or "database" in title_lower:
            constraints.append(
                "Create Alembic migration, never modify existing migrations"
            )

        # Universal rules — always present.
        constraints += [
            "Format code with ruff before committing",
            "Add type hints to all new functions",
            "Update inline documentation for any modified public methods",
            "Never hardcode credentials or API keys",
        ]

        return constraints

    def _estimate_files_to_modify(
        self, context_files: list[CodeContext], task: TechnicalTask
    ) -> list[str]:
        """Predicts which file paths are most likely to be modified or created."""
        return [f.file_path for f in context_files[:3]]

    def _build_llm_prompt(
        self,
        task: TechnicalTask,
        context: list[CodeContext],
        memory: MemoryContent,
        feedback: str | None = None,
    ) -> str:
        """Builds the LLM meta-prompt that asks the model to write coding instructions."""
        context_block = "\n\n".join(
            f"### {c.file_path} ({c.language})\n```{c.language}\n{c.content}\n```"
            for c in context
        )
        patterns_text = (
            "\n".join(
                f"  - {p.name}: {p.description}" for p in memory.naming_patterns
            )
            or "  None recorded."
        )
        decisions_text = (
            "\n".join(
                f"  - {d.decision}: {d.rationale}"
                for d in memory.architectural_decisions
            )
            or "  None recorded."
        )
        feedback_section = (
            f"\n## Refinement Feedback\n{feedback}\nRevise your output accordingly.\n"
            if feedback
            else ""
        )
        schema_example = json.dumps(
            {
                "task_id": task.id,
                "task_title": task.title,
                "instructions": "Step-by-step implementation instructions (>=200 chars)",
                "context_files": [],
                "stack": "",
                "constraints": [],
                "expected_output": "What the Vibe Coder must produce",
                "acceptance_criteria": [],
                "naming_patterns": [],
                "estimated_files_to_modify": ["path/to/file.py"],
                "created_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        )

        return f"""You are an expert prompt engineer preparing coding instructions for an AI coding agent.

## Task to implement
ID: {task.id}
Title: {task.title}
Description: {task.description}
Story points: {task.story_points}
Assignee role: {task.suggested_assignee or "unspecified"}

## Codebase Context (RAG-retrieved files)
{context_block or "No context files available — write general instructions."}

## Product Memory
### Naming Patterns
{patterns_text}

### Architectural Decisions
{decisions_text}
{feedback_section}
## Instructions
1. Read the task description and the codebase context files carefully.
2. Write precise, numbered, step-by-step implementation instructions for a coding agent.
3. Reference specific existing functions, classes, or patterns from the context files.
4. Specify exactly which files to create or modify.
5. Specify the expected structure of any new functions or classes (signatures, return types).
6. Flag any ambiguities or edge cases the coder must handle.
7. Fill `estimated_files_to_modify` with the predicted file paths.
8. Fill `expected_output` with a clear description of the deliverable.

## Output Format
Return ONLY a valid JSON object. No markdown, no explanation.
Schema:
{schema_example}

IMPORTANT: `instructions` must be at least 200 characters with numbered steps.
"""

    def _parse_llm_response(
        self,
        raw: str,
        task: TechnicalTask,
        context: list[CodeContext],
    ) -> VibeCoderPrompt:
        """Extracts and deserialises the JSON VibeCoderPrompt from the LLM response."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        data: dict = json.loads(text)
        # Always ensure task_id and task_title match the actual task.
        data["task_id"] = task.id
        data["task_title"] = task.title
        return VibeCoderPrompt.model_validate(data)

    def _validate_prompt(self, prompt: VibeCoderPrompt) -> None:
        """Logs warnings for quality issues; does not raise."""
        if len(prompt.instructions) < _MIN_INSTRUCTIONS_LEN:
            logger.warning(
                "Prompt for %s has short instructions (%d chars < %d required). "
                "Proceeding with best effort.",
                prompt.task_id,
                len(prompt.instructions),
                _MIN_INSTRUCTIONS_LEN,
            )
        if not prompt.context_files:
            logger.warning(
                "Prompt for %s has no context files — RAG index may be empty.",
                prompt.task_id,
            )
        if not prompt.expected_output:
            logger.warning("Prompt for %s has empty expected_output.", prompt.task_id)
        if not prompt.acceptance_criteria:
            logger.warning(
                "Prompt for %s has no acceptance criteria.", prompt.task_id
            )

    @staticmethod
    def _infer_reason(file_path: str, task: TechnicalTask) -> str:
        """Returns a short string explaining why this file is relevant to the task."""
        name = file_path.split("/")[-1].lower()
        title_words = set(task.title.lower().split())
        if any(w in name for w in title_words):
            return f"Filename matches task keywords from '{task.title}'"
        return f"Semantically similar to task: {task.description[:80]}"

    @staticmethod
    def _get_acceptance_criteria(
        task: TechnicalTask, sprint_plan: SprintPlan
    ) -> list[AcceptanceCriteria]:
        """Finds acceptance criteria from the user story linked to this task."""
        # The acceptance criteria live on user stories in the PRD, not the sprint plan.
        # We return an empty list here; the caller can enrich if PRD is available.
        return []


# ---------------------------------------------------------------------------
# CrewAI wrapper
# ---------------------------------------------------------------------------


class PromptEngineerAgent:
    """CrewAI Agent wrapper for the PromptEngineer.

    The engineering logic lives in PromptEngineer. This class exposes a
    build() method returning a configured crewai.Agent for use in ProdutoCrew.
    """

    def build(self) -> Agent:
        """Returns a configured CrewAI Agent for the prompt-engineering role."""
        return Agent(
            role="Prompt Engineer",
            goal=(
                "Convert approved technical tasks into precise, context-rich prompts "
                "for the Vibe Coder using RAG to automatically include relevant "
                "codebase files."
            ),
            backstory=(
                "You are an expert at translating technical requirements into "
                "unambiguous coding instructions. You always search the codebase for "
                "relevant context before writing a prompt. You reference specific "
                "existing files and patterns so the Vibe Coder never writes "
                "inconsistent code. You make constraints explicit so nothing is left "
                "to interpretation."
            ),
            llm=_LLM_MODEL,
            memory=True,
            max_iter=3,
            verbose=True,
        )
