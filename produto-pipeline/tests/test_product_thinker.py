"""Unit tests for agents/product_thinker.py — ProductThinker and its models."""

import json
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from agents.ceo_orchestrator import FounderRequest
from agents.product_thinker import (
    AcceptanceCriteria,
    CheckpointError,
    ConflictError,
    PRDOutput,
    ProductThinker,
    ThinkerInput,
    UserStory,
)
from context.memory import (
    DiscardedFeature,
    ImplementedFeature,
    MemoryContent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 15, 10, 0, 0)


def _make_story(id_: str = "US-001", title: str = "Login") -> UserStory:
    return UserStory(
        id=id_,
        title=title,
        as_a="user",
        i_want="to log in",
        so_that="I can access my account",
        acceptance_criteria=[
            AcceptanceCriteria(
                given="I am on the login page",
                when="I submit valid credentials",
                then="I am redirected to the dashboard",
            )
        ],
    )


def _make_prd(feature_name: str = "Auth System") -> PRDOutput:
    return PRDOutput(
        feature_name=feature_name,
        problem_statement="Users cannot log in",
        proposed_solution="Implement JWT authentication",
        out_of_scope=["SSO", "OAuth"],
        user_stories=[_make_story("US-001"), _make_story("US-002", "Logout")],
        open_questions=["Should we support magic links?"],
        created_at=_NOW,
    )


def _make_memory(
    implemented: list[ImplementedFeature] | None = None,
    discarded: list[DiscardedFeature] | None = None,
) -> MemoryContent:
    return MemoryContent(
        implemented_features=implemented or [],
        discarded_features=discarded or [],
    )


def _make_thinker_input(
    problem: str = "Build an auth system",
    memory: MemoryContent | None = None,
) -> ThinkerInput:
    return ThinkerInput(
        request=FounderRequest(
            problem=problem,
            stack="FastAPI + React",
            repo="org/repo",
            priority="normal",
            requested_at=_NOW,
        ),
        memory=memory or _make_memory(),
    )


def _make_thinker() -> tuple[ProductThinker, MagicMock]:
    """Returns a ProductThinker with a mocked Anthropic client."""
    mock_client = MagicMock()
    thinker = ProductThinker(llm_client=mock_client)
    return thinker, mock_client


def _stub_llm(mock_client: MagicMock, prd: PRDOutput) -> None:
    """Configures the mock LLM client to return a serialised PRDOutput."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=prd.model_dump_json())]
    mock_client.messages.create.return_value = mock_message


# ---------------------------------------------------------------------------
# _check_conflicts()
# ---------------------------------------------------------------------------


class TestCheckConflicts:
    def test_no_conflicts_with_new_feature(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd("Dark Mode")
        memory = _make_memory(
            implemented=[ImplementedFeature(name="Auth", description="JWT", date=_NOW)]
        )

        conflicts = thinker._check_conflicts(prd, memory)

        assert conflicts == []

    def test_conflict_with_implemented_feature(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd("Auth System")
        memory = _make_memory(
            implemented=[
                ImplementedFeature(name="Auth System", description="JWT", date=_NOW)
            ]
        )

        conflicts = thinker._check_conflicts(prd, memory)

        assert len(conflicts) == 1
        assert "Auth System" in conflicts[0]
        assert "implemented" in conflicts[0].lower()

    def test_conflict_with_discarded_feature(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd("Dark Mode")
        memory = _make_memory(
            discarded=[
                DiscardedFeature(name="Dark Mode", reason="Low ROI", date=_NOW)
            ]
        )

        conflicts = thinker._check_conflicts(prd, memory)

        assert len(conflicts) == 1
        assert "Dark Mode" in conflicts[0]
        assert "discarded" in conflicts[0].lower()

    def test_conflict_with_both_implemented_and_discarded(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd("Auth System")
        memory = _make_memory(
            implemented=[
                ImplementedFeature(name="Auth System", description="JWT", date=_NOW)
            ],
            discarded=[
                DiscardedFeature(name="Auth System", reason="Old reason", date=_NOW)
            ],
        )

        conflicts = thinker._check_conflicts(prd, memory)

        assert len(conflicts) == 2

    def test_partial_name_match_detects_conflict(self) -> None:
        thinker, _ = _make_thinker()
        # "Auth" is contained in "Auth System"
        prd = _make_prd("Auth System")
        memory = _make_memory(
            implemented=[
                ImplementedFeature(name="Auth", description="JWT", date=_NOW)
            ]
        )

        conflicts = thinker._check_conflicts(prd, memory)

        assert len(conflicts) == 1

    def test_empty_memory_returns_no_conflicts(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd("Brand New Feature")
        memory = _make_memory()

        conflicts = thinker._check_conflicts(prd, memory)

        assert conflicts == []


# ---------------------------------------------------------------------------
# run() — conflict handling
# ---------------------------------------------------------------------------


class TestRunConflict:
    def test_run_raises_conflict_error_when_feature_implemented(self) -> None:
        thinker, mock_client = _make_thinker()
        prd = _make_prd("Auth System")
        _stub_llm(mock_client, prd)

        memory = _make_memory(
            implemented=[
                ImplementedFeature(name="Auth System", description="JWT", date=_NOW)
            ]
        )
        thinker_input = _make_thinker_input(memory=memory)

        with pytest.raises(ConflictError) as exc_info:
            thinker.run(thinker_input)

        assert "Auth System" in str(exc_info.value)
        assert len(exc_info.value.conflicts) > 0

    def test_run_raises_conflict_error_when_feature_discarded(self) -> None:
        thinker, mock_client = _make_thinker()
        prd = _make_prd("Dark Mode")
        _stub_llm(mock_client, prd)

        memory = _make_memory(
            discarded=[
                DiscardedFeature(name="Dark Mode", reason="Low ROI", date=_NOW)
            ]
        )
        thinker_input = _make_thinker_input(memory=memory)

        with pytest.raises(ConflictError):
            thinker.run(thinker_input)

    def test_run_succeeds_with_no_conflicts(self) -> None:
        thinker, mock_client = _make_thinker()
        prd = _make_prd("Brand New Feature")
        _stub_llm(mock_client, prd)

        result = thinker.run(_make_thinker_input())

        assert result.feature_name == "Brand New Feature"

    def test_run_calls_llm_with_prompt(self) -> None:
        thinker, mock_client = _make_thinker()
        prd = _make_prd("New Feature")
        _stub_llm(mock_client, prd)

        thinker_input = _make_thinker_input()
        thinker.run(thinker_input)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert len(call_kwargs["messages"]) == 1


# ---------------------------------------------------------------------------
# checkpoint() — approve path
# ---------------------------------------------------------------------------


class TestCheckpointApprove:
    def test_approve_returns_prd_unchanged(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="approve")
        original_input = _make_thinker_input()

        result = thinker.checkpoint(prd, notify_fn, poll_fn, original_input)

        assert result is prd

    def test_approve_sends_prd_summary_to_slack(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="approve")

        thinker.checkpoint(prd, notify_fn, poll_fn, _make_thinker_input())

        notify_fn.assert_called_once()
        msg = notify_fn.call_args[0][0]
        assert prd.feature_name in msg

    def test_approve_case_insensitive(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="  APPROVE  ")

        result = thinker.checkpoint(prd, notify_fn, poll_fn, _make_thinker_input())

        assert result is prd


# ---------------------------------------------------------------------------
# checkpoint() — reject + re-run path
# ---------------------------------------------------------------------------


class TestCheckpointReject:
    def test_reject_reruns_with_feedback(self) -> None:
        thinker, mock_client = _make_thinker()
        initial_prd = _make_prd("Auth System")
        revised_prd = _make_prd("Auth System v2")

        # First call: initial PRD; second call: revised PRD after rejection.
        mock_message_1 = MagicMock()
        mock_message_1.content = [MagicMock(text=initial_prd.model_dump_json())]
        mock_message_2 = MagicMock()
        mock_message_2.content = [MagicMock(text=revised_prd.model_dump_json())]
        mock_client.messages.create.side_effect = [mock_message_1, mock_message_2]

        notify_fn = MagicMock()
        # First poll: reject with feedback; second poll: approve.
        poll_fn = MagicMock(side_effect=["reject: add more detail", "approve"])
        original_input = _make_thinker_input()

        result = thinker.checkpoint(initial_prd, notify_fn, poll_fn, original_input)

        # Should have called LLM again for the re-run.
        assert mock_client.messages.create.call_count == 1  # Only during re-run
        # Final approved PRD is the revised one.
        assert result.feature_name == "Auth System v2"

    def test_reject_feedback_included_in_rerun_prompt(self) -> None:
        thinker, mock_client = _make_thinker()
        initial_prd = _make_prd()
        revised_prd = _make_prd("Revised Feature")

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=revised_prd.model_dump_json())]
        mock_client.messages.create.return_value = mock_message

        notify_fn = MagicMock()
        poll_fn = MagicMock(side_effect=["reject: needs more user stories", "approve"])

        thinker.checkpoint(initial_prd, notify_fn, poll_fn, _make_thinker_input())

        # The prompt sent to the LLM during re-run must contain the feedback.
        rerun_prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        assert "needs more user stories" in rerun_prompt

    def test_unrecognised_reply_prompts_clarification_without_consuming_attempt(
        self,
    ) -> None:
        thinker, mock_client = _make_thinker()
        prd = _make_prd()
        notify_fn = MagicMock()
        # Unrecognised → clarification; then approve.
        poll_fn = MagicMock(side_effect=["yes please", "approve"])

        result = thinker.checkpoint(prd, notify_fn, poll_fn, _make_thinker_input())

        assert result is prd
        # Should have posted clarification message.
        clarification_calls = [
            c for c in notify_fn.call_args_list if "Please reply with" in c[0][0]
        ]
        assert len(clarification_calls) == 1
        # LLM should NOT have been called (no re-run for unrecognised reply).
        mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# checkpoint() — max retries exhausted
# ---------------------------------------------------------------------------


class TestCheckpointMaxRetries:
    def test_raises_checkpoint_error_after_3_rejections(self) -> None:
        thinker, mock_client = _make_thinker()
        initial_prd = _make_prd()
        revised_prd = _make_prd("Revised")

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=revised_prd.model_dump_json())]
        mock_client.messages.create.return_value = mock_message

        notify_fn = MagicMock()
        poll_fn = MagicMock(
            side_effect=[
                "reject: not good enough",
                "reject: still not good",
                "reject: I give up",
            ]
        )

        with pytest.raises(CheckpointError) as exc_info:
            thinker.checkpoint(initial_prd, notify_fn, poll_fn, _make_thinker_input())

        assert "3" in str(exc_info.value)
        assert exc_info.value.prd is not None

    def test_checkpoint_error_carries_last_prd(self) -> None:
        thinker, mock_client = _make_thinker()
        initial_prd = _make_prd("Initial")
        last_prd = _make_prd("Last Attempt")

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=last_prd.model_dump_json())]
        mock_client.messages.create.return_value = mock_message

        notify_fn = MagicMock()
        poll_fn = MagicMock(
            side_effect=["reject: 1", "reject: 2", "reject: 3"]
        )

        with pytest.raises(CheckpointError) as exc_info:
            thinker.checkpoint(initial_prd, notify_fn, poll_fn, _make_thinker_input())

        assert exc_info.value.prd is not None
        assert exc_info.value.prd.feature_name == "Last Attempt"

    def test_raises_checkpoint_error_on_timeout(self) -> None:
        thinker, _ = _make_thinker()
        prd = _make_prd()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value=None)  # timeout

        with pytest.raises(CheckpointError) as exc_info:
            thinker.checkpoint(prd, notify_fn, poll_fn, _make_thinker_input())

        assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# _parse_response()
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _raw_prd_json(self) -> str:
        return _make_prd().model_dump_json()

    def test_parses_plain_json(self) -> None:
        thinker, _ = _make_thinker()
        result = thinker._parse_response(self._raw_prd_json())
        assert isinstance(result, PRDOutput)

    def test_parses_json_in_markdown_code_fence(self) -> None:
        thinker, _ = _make_thinker()
        wrapped = f"```json\n{self._raw_prd_json()}\n```"
        result = thinker._parse_response(wrapped)
        assert isinstance(result, PRDOutput)

    def test_parses_json_in_generic_code_fence(self) -> None:
        thinker, _ = _make_thinker()
        wrapped = f"```\n{self._raw_prd_json()}\n```"
        result = thinker._parse_response(wrapped)
        assert isinstance(result, PRDOutput)

    def test_raises_on_invalid_json(self) -> None:
        thinker, _ = _make_thinker()
        with pytest.raises(Exception):
            thinker._parse_response("not json at all")
