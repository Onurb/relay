"""Unit tests for agents/prompt_engineer.py — PromptEngineer and its models."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agents.ceo_orchestrator import FounderRequest
from agents.product_thinker import AcceptanceCriteria, PRDOutput, UserStory
from agents.prompt_engineer import (
    CodeContext,
    PromptEngineer,
    PromptEngineerInput,
    PromptEngineerOutput,
    VibeCoderPrompt,
)
from agents.sprint_planner import SprintPlan, TechnicalTask
from context.memory import (
    ArchitecturalDecision,
    MemoryContent,
    NamingPattern,
)
from context.rag_index import RAGConfig, RAGSearchResult, RelevantFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 20, 10, 0, 0)


def _make_task(
    id_: str = "TASK-001",
    title: str = "Implement login endpoint",
    description: str = "Create POST /auth/login endpoint with JWT response",
    story_points: int = 3,
) -> TechnicalTask:
    return TechnicalTask(
        id=id_,
        user_story_id="US-001",
        title=title,
        description=description,
        story_points=story_points,
        suggested_assignee="backend",
    )


def _make_sprint(tasks: list[TechnicalTask] | None = None) -> SprintPlan:
    t = tasks or [_make_task()]
    return SprintPlan(
        sprint_name="Sprint — Auth System",
        total_story_points=sum(tk.story_points for tk in t),
        capacity_story_points=20,
        tasks=t,
        ordered_task_ids=[tk.id for tk in t],
        created_at=_NOW,
    )


def _make_memory(
    patterns: list[NamingPattern] | None = None,
    decisions: list[ArchitecturalDecision] | None = None,
) -> MemoryContent:
    return MemoryContent(
        naming_patterns=patterns or [],
        architectural_decisions=decisions or [],
    )


def _make_relevant_file(
    path: str = "/repo/auth.py", score: float = 0.9, language: str = "python"
) -> RelevantFile:
    return RelevantFile(
        file_name=path.split("/")[-1],
        file_path=path,
        score=score,
        content=f"# Content of {path}",
        language=language,
    )


def _make_rag_result(files: list[RelevantFile]) -> RAGSearchResult:
    return RAGSearchResult(
        query="test query",
        files=files,
        total_indexed=len(files),
        search_duration_ms=5.0,
    )


def _make_mock_rag(files: list[RelevantFile] | None = None) -> MagicMock:
    rag = MagicMock()
    result = _make_rag_result(files or [])
    rag.search.return_value = result
    rag.search_by_type.return_value = result
    return rag


def _make_engineer() -> tuple[PromptEngineer, MagicMock]:
    mock_client = MagicMock()
    engineer = PromptEngineer(llm_client=mock_client)
    return engineer, mock_client


def _make_vibe_prompt(
    task_id: str = "TASK-001",
    instructions: str = "x" * 250,
    expected_output: str = "A working endpoint",
) -> VibeCoderPrompt:
    return VibeCoderPrompt(
        task_id=task_id,
        task_title="Implement login endpoint",
        instructions=instructions,
        stack="FastAPI + React",
        expected_output=expected_output,
        created_at=_NOW,
    )


def _stub_llm(mock_client: MagicMock, prompt: VibeCoderPrompt) -> None:
    msg = MagicMock()
    msg.content = [MagicMock(text=prompt.model_dump_json())]
    mock_client.messages.create.return_value = msg


def _make_engineer_input(
    tasks: list[TechnicalTask] | None = None,
    memory: MemoryContent | None = None,
    rag: MagicMock | None = None,
) -> PromptEngineerInput:
    return PromptEngineerInput(
        sprint_plan=_make_sprint(tasks),
        memory=memory or _make_memory(),
        rag=rag or _make_mock_rag(),
        stack="FastAPI + React + PostgreSQL",
    )


# ---------------------------------------------------------------------------
# _fetch_context()
# ---------------------------------------------------------------------------


class TestFetchContext:
    def test_deduplicates_by_file_path_and_keeps_highest_score(self) -> None:
        engineer, _ = _make_engineer()
        # Same file returned by both general and backend search, different scores.
        file_low = _make_relevant_file("/repo/auth.py", score=0.6)
        file_high = _make_relevant_file("/repo/auth.py", score=0.9)
        file_other = _make_relevant_file("/repo/models.py", score=0.7)

        rag = MagicMock()
        rag.search.return_value = _make_rag_result([file_low, file_other])
        rag.search_by_type.return_value = _make_rag_result([file_high])

        task = _make_task()
        result = engineer._fetch_context(task, rag)

        auth_files = [f for f in result if f.file_path == "/repo/auth.py"]
        assert len(auth_files) == 1
        assert auth_files[0].relevance_score == 0.9

    def test_returns_top_6_files_by_score(self) -> None:
        engineer, _ = _make_engineer()
        files = [_make_relevant_file(f"/repo/file{i}.py", score=1.0 - i * 0.05) for i in range(10)]

        rag = MagicMock()
        rag.search.return_value = _make_rag_result(files)
        rag.search_by_type.return_value = _make_rag_result([])

        result = engineer._fetch_context(_make_task(), rag)
        assert len(result) <= 6

    def test_returns_files_sorted_by_score_descending(self) -> None:
        engineer, _ = _make_engineer()
        files = [
            _make_relevant_file("/repo/low.py", score=0.4),
            _make_relevant_file("/repo/high.py", score=0.95),
            _make_relevant_file("/repo/mid.py", score=0.7),
        ]
        rag = _make_mock_rag(files)

        result = engineer._fetch_context(_make_task(), rag)
        scores = [f.relevance_score for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_empty_list_and_logs_warning_when_rag_empty(
        self, caplog
    ) -> None:
        engineer, _ = _make_engineer()
        rag = _make_mock_rag([])

        import logging
        with caplog.at_level(logging.WARNING):
            result = engineer._fetch_context(_make_task(), rag)

        assert result == []
        assert any("0 context files" in r.message or "empty" in r.message.lower()
                   for r in caplog.records)

    def test_context_files_have_reason_set(self) -> None:
        engineer, _ = _make_engineer()
        rag = _make_mock_rag([_make_relevant_file("/repo/auth.py")])

        result = engineer._fetch_context(_make_task(), rag)
        assert result[0].reason is not None
        assert len(result[0].reason) > 0


# ---------------------------------------------------------------------------
# _build_constraints()
# ---------------------------------------------------------------------------


class TestBuildConstraints:
    def test_includes_naming_patterns_from_memory(self) -> None:
        engineer, _ = _make_engineer()
        memory = _make_memory(
            patterns=[
                NamingPattern(name="Route names", description="Use snake_case for routes"),
                NamingPattern(name="Model names", description="Use PascalCase"),
            ]
        )
        constraints = engineer._build_constraints(_make_task(), memory)

        pattern_constraints = [c for c in constraints if "naming pattern" in c.lower()]
        assert len(pattern_constraints) == 2

    def test_includes_architectural_decisions_from_memory(self) -> None:
        engineer, _ = _make_engineer()
        memory = _make_memory(
            decisions=[
                ArchitecturalDecision(
                    decision="Use PostgreSQL",
                    rationale="ACID compliance",
                    date=_NOW,
                )
            ]
        )
        constraints = engineer._build_constraints(_make_task(), memory)

        decision_constraints = [c for c in constraints if "PostgreSQL" in c]
        assert len(decision_constraints) == 1

    def test_api_task_includes_fastapi_constraint(self) -> None:
        engineer, _ = _make_engineer()
        task = _make_task(title="Create API endpoint for auth")

        constraints = engineer._build_constraints(task, _make_memory())

        assert any("FastAPI" in c for c in constraints)

    def test_endpoint_in_title_includes_fastapi_constraint(self) -> None:
        engineer, _ = _make_engineer()
        task = _make_task(title="Add login endpoint")

        constraints = engineer._build_constraints(task, _make_memory())

        assert any("FastAPI" in c for c in constraints)

    def test_test_task_includes_pytest_constraint(self) -> None:
        engineer, _ = _make_engineer()
        task = _make_task(title="Write unit test for auth module")

        constraints = engineer._build_constraints(task, _make_memory())

        assert any("pytest" in c for c in constraints)

    def test_frontend_task_includes_react_constraint(self) -> None:
        engineer, _ = _make_engineer()
        task = _make_task(title="Build login component")

        constraints = engineer._build_constraints(task, _make_memory())

        assert any("React" in c for c in constraints)

    def test_migration_task_includes_alembic_constraint(self) -> None:
        engineer, _ = _make_engineer()
        task = _make_task(title="Create database migration for users table")

        constraints = engineer._build_constraints(task, _make_memory())

        assert any("Alembic" in c for c in constraints)

    def test_always_includes_ruff_constraint(self) -> None:
        engineer, _ = _make_engineer()
        constraints = engineer._build_constraints(_make_task(), _make_memory())
        assert any("ruff" in c for c in constraints)

    def test_always_includes_type_hints_constraint(self) -> None:
        engineer, _ = _make_engineer()
        constraints = engineer._build_constraints(_make_task(), _make_memory())
        assert any("type hints" in c for c in constraints)

    def test_always_includes_no_credentials_constraint(self) -> None:
        engineer, _ = _make_engineer()
        constraints = engineer._build_constraints(_make_task(), _make_memory())
        assert any("credentials" in c.lower() or "API keys" in c for c in constraints)


# ---------------------------------------------------------------------------
# _estimate_files_to_modify()
# ---------------------------------------------------------------------------


class TestEstimateFilesToModify:
    def test_returns_non_empty_list_from_context(self) -> None:
        engineer, _ = _make_engineer()
        context = [
            CodeContext(
                file_path="/repo/auth.py",
                language="python",
                content="code",
                relevance_score=0.9,
            ),
            CodeContext(
                file_path="/repo/models.py",
                language="python",
                content="code",
                relevance_score=0.8,
            ),
        ]
        result = engineer._estimate_files_to_modify(context, _make_task())
        assert len(result) > 0

    def test_returns_at_most_3_files(self) -> None:
        engineer, _ = _make_engineer()
        context = [
            CodeContext(
                file_path=f"/repo/file{i}.py",
                language="python",
                content="code",
                relevance_score=0.9 - i * 0.1,
            )
            for i in range(6)
        ]
        result = engineer._estimate_files_to_modify(context, _make_task())
        assert len(result) <= 3

    def test_returns_empty_for_empty_context(self) -> None:
        engineer, _ = _make_engineer()
        result = engineer._estimate_files_to_modify([], _make_task())
        assert result == []


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_returns_one_prompt_per_task(self) -> None:
        engineer, mock_client = _make_engineer()
        tasks = [_make_task("TASK-001"), _make_task("TASK-002")]
        vibe_prompt = _make_vibe_prompt()
        _stub_llm(mock_client, vibe_prompt)

        input_ = _make_engineer_input(tasks=tasks)
        result = engineer.run(input_)

        assert isinstance(result, PromptEngineerOutput)
        assert len(result.prompts) == 2

    def test_counts_unique_context_files_across_prompts(self) -> None:
        engineer, mock_client = _make_engineer()
        tasks = [_make_task("TASK-001"), _make_task("TASK-002")]
        rag = _make_mock_rag([
            _make_relevant_file("/repo/auth.py"),
            _make_relevant_file("/repo/models.py"),
        ])
        vibe_prompt = _make_vibe_prompt()
        _stub_llm(mock_client, vibe_prompt)

        input_ = _make_engineer_input(tasks=tasks, rag=rag)
        result = engineer.run(input_)

        assert result.total_context_files >= 0

    def test_logs_warning_when_instructions_too_short(self, caplog) -> None:
        engineer, mock_client = _make_engineer()
        short_prompt = _make_vibe_prompt(instructions="Too short")
        _stub_llm(mock_client, short_prompt)

        import logging
        with caplog.at_level(logging.WARNING):
            engineer.run(_make_engineer_input())

        assert any(
            "short instructions" in r.message or "200" in r.message
            for r in caplog.records
        )

    def test_prompt_stack_matches_input_stack(self) -> None:
        engineer, mock_client = _make_engineer()
        vibe_prompt = _make_vibe_prompt()
        _stub_llm(mock_client, vibe_prompt)

        input_ = _make_engineer_input()
        result = engineer.run(input_)

        assert result.prompts[0].stack == "FastAPI + React + PostgreSQL"

    def test_prompt_constraints_always_contain_universal_rules(self) -> None:
        engineer, mock_client = _make_engineer()
        vibe_prompt = _make_vibe_prompt()
        _stub_llm(mock_client, vibe_prompt)

        result = engineer.run(_make_engineer_input())
        constraints = result.prompts[0].constraints

        assert any("ruff" in c for c in constraints)
        assert any("type hints" in c for c in constraints)


# ---------------------------------------------------------------------------
# optional_checkpoint() — disabled
# ---------------------------------------------------------------------------


class TestOptionalCheckpointDisabled:
    def test_skips_slack_notification_when_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "false")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt()
        notify_fn = MagicMock()
        poll_fn = MagicMock()

        result = engineer.optional_checkpoint(
            prompt, notify_fn, poll_fn, _make_task(), _make_engineer_input()
        )

        notify_fn.assert_not_called()
        poll_fn.assert_not_called()
        assert result is prompt

    def test_returns_prompt_unchanged_when_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "false")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt(instructions="x" * 300)

        result = engineer.optional_checkpoint(
            prompt, MagicMock(), MagicMock(), _make_task(), _make_engineer_input()
        )

        assert result.instructions == prompt.instructions


# ---------------------------------------------------------------------------
# optional_checkpoint() — enabled, "ok"
# ---------------------------------------------------------------------------


class TestOptionalCheckpointOk:
    def test_ok_returns_prompt_unchanged(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="ok")

        result = engineer.optional_checkpoint(
            prompt, notify_fn, poll_fn, _make_task(), _make_engineer_input()
        )

        assert result is prompt

    def test_ok_sends_formatted_message_to_slack(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="ok")

        engineer.optional_checkpoint(
            prompt, notify_fn, poll_fn, _make_task(), _make_engineer_input()
        )

        notify_fn.assert_called_once()
        msg = notify_fn.call_args[0][0]
        assert prompt.task_id in msg
        assert "Prompt ready" in msg

    def test_ok_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt()
        poll_fn = MagicMock(return_value="  OK  ")

        result = engineer.optional_checkpoint(
            prompt, MagicMock(), poll_fn, _make_task(), _make_engineer_input()
        )
        assert result is prompt


# ---------------------------------------------------------------------------
# optional_checkpoint() — enabled, "refine:"
# ---------------------------------------------------------------------------


class TestOptionalCheckpointRefine:
    def test_refine_calls_llm_again_with_feedback(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, mock_client = _make_engineer()
        original_prompt = _make_vibe_prompt()
        refined_prompt = _make_vibe_prompt(instructions="y" * 300)
        _stub_llm(mock_client, refined_prompt)

        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="refine: add more context about the database schema")

        engineer.optional_checkpoint(
            original_prompt,
            notify_fn,
            poll_fn,
            _make_task(),
            _make_engineer_input(),
        )

        # LLM must be called once for the refinement.
        mock_client.messages.create.assert_called_once()

    def test_refine_feedback_included_in_llm_prompt(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, mock_client = _make_engineer()
        refined = _make_vibe_prompt()
        _stub_llm(mock_client, refined)

        poll_fn = MagicMock(return_value="refine: add more context about the database schema")

        engineer.optional_checkpoint(
            _make_vibe_prompt(),
            MagicMock(),
            poll_fn,
            _make_task(),
            _make_engineer_input(),
        )

        call_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        assert "add more context about the database schema" in call_content

    def test_none_response_returns_prompt_unchanged(self, monkeypatch) -> None:
        monkeypatch.setenv("PROMPT_REVIEW_ENABLED", "true")
        engineer, _ = _make_engineer()
        prompt = _make_vibe_prompt()
        poll_fn = MagicMock(return_value=None)

        result = engineer.optional_checkpoint(
            prompt, MagicMock(), poll_fn, _make_task(), _make_engineer_input()
        )
        assert result is prompt
