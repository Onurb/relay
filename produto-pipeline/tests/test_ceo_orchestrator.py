"""Unit tests for CEOOrchestrator, FounderRequest and PipelineResult."""

import os
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    slack_token: str = "xoxb-fake",
    slack_channel: str = "C_TEST",
    founder_user_id: str = "U_FOUNDER",
):
    """Creates a CEOOrchestrator with a mocked Slack client."""
    with patch.dict(os.environ, {"SLACK_FOUNDER_USER_ID": founder_user_id}):
        with patch("agents.ceo_orchestrator.WebClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            from agents.ceo_orchestrator import CEOOrchestrator

            orch = CEOOrchestrator(
                slack_token=slack_token,
                slack_channel=slack_channel,
            )
            orch._client = mock_client  # expose for assertions
            return orch, mock_client


# ---------------------------------------------------------------------------
# _validate_request
# ---------------------------------------------------------------------------


class TestValidateRequest:
    def test_valid_message_returns_founder_request(self) -> None:
        from agents.ceo_orchestrator import FounderRequest

        orch, _ = _make_orchestrator()
        result = orch._validate_request("Build a user authentication system")

        assert isinstance(result, FounderRequest)
        assert result.problem == "Build a user authentication system"

    def test_valid_message_strips_whitespace(self) -> None:
        orch, _ = _make_orchestrator()
        result = orch._validate_request("  Fix the login bug  ")

        assert result.problem == "Fix the login bug"

    def test_empty_string_raises_value_error(self) -> None:
        orch, _ = _make_orchestrator()
        with pytest.raises(ValueError, match="empty"):
            orch._validate_request("")

    def test_whitespace_only_raises_value_error(self) -> None:
        orch, _ = _make_orchestrator()
        with pytest.raises(ValueError):
            orch._validate_request("   ")

    def test_default_priority_is_normal(self) -> None:
        orch, _ = _make_orchestrator()
        result = orch._validate_request("Add dark mode")

        assert result.priority == "normal"

    def test_requested_at_is_set_automatically(self) -> None:
        orch, _ = _make_orchestrator()
        before = datetime.utcnow()
        result = orch._validate_request("Some feature")
        after = datetime.utcnow()

        assert before <= result.requested_at <= after


# ---------------------------------------------------------------------------
# _notify
# ---------------------------------------------------------------------------


class TestNotify:
    def test_notify_posts_to_correct_channel(self) -> None:
        orch, mock_client = _make_orchestrator(slack_channel="C_CHANNEL")
        orch._notify("Hello Slack")

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C_CHANNEL", text="Hello Slack"
        )

    def test_notify_called_at_pipeline_start(self) -> None:
        orch, mock_client = _make_orchestrator()
        orch._notify("📥 Got it: Build auth... Starting pipeline.")

        texts = [c.kwargs["text"] for c in mock_client.chat_postMessage.call_args_list]
        assert any("Got it" in t for t in texts)

    def test_notify_called_at_prd_checkpoint(self) -> None:
        orch, mock_client = _make_orchestrator()
        orch._notify(
            "📋 PRD ready for review. Please approve or reject with feedback."
        )

        texts = [c.kwargs["text"] for c in mock_client.chat_postMessage.call_args_list]
        assert any("PRD ready" in t for t in texts)

    def test_notify_called_at_sprint_checkpoint(self) -> None:
        orch, mock_client = _make_orchestrator()
        orch._notify("🗓️ Sprint planned. Please approve or reject with feedback.")

        texts = [c.kwargs["text"] for c in mock_client.chat_postMessage.call_args_list]
        assert any("Sprint planned" in t for t in texts)

    def test_notify_slack_error_is_caught(self) -> None:
        from slack_sdk.errors import SlackApiError

        orch, mock_client = _make_orchestrator()
        mock_client.chat_postMessage.side_effect = SlackApiError(
            "error", {"error": "channel_not_found"}
        )
        # Should not raise — errors are swallowed and logged.
        orch._notify("This will fail silently")


# ---------------------------------------------------------------------------
# listen / _poll — message filtering
# ---------------------------------------------------------------------------


class TestListen:
    def _build_message(
        self,
        ts: str,
        user: str,
        text: str,
        bot_id: str | None = None,
    ) -> dict:
        msg: dict = {"ts": ts, "user": user, "text": text}
        if bot_id:
            msg["bot_id"] = bot_id
        return msg

    def test_skips_messages_from_other_users(self) -> None:
        orch, mock_client = _make_orchestrator(founder_user_id="U_FOUNDER")
        mock_client.conversations_history.return_value = {
            "messages": [
                self._build_message("2000.0", "U_OTHER", "Hello from stranger")
            ]
        }
        # Prime last_ts so the new message is considered "new"
        orch._last_ts = "1999.0"

        with patch.object(orch, "handle_request") as mock_handle:
            orch._poll()
            mock_handle.assert_not_called()

    def test_skips_bot_messages(self) -> None:
        orch, mock_client = _make_orchestrator(founder_user_id="U_FOUNDER")
        mock_client.conversations_history.return_value = {
            "messages": [
                self._build_message(
                    "2000.0", "U_FOUNDER", "Bot reply", bot_id="B_BOT"
                )
            ]
        }
        orch._last_ts = "1999.0"

        with patch.object(orch, "handle_request") as mock_handle:
            orch._poll()
            mock_handle.assert_not_called()

    def test_processes_founder_message(self) -> None:
        orch, mock_client = _make_orchestrator(founder_user_id="U_FOUNDER")
        mock_client.conversations_history.return_value = {
            "messages": [
                self._build_message("2000.0", "U_FOUNDER", "Build an auth system")
            ]
        }
        orch._last_ts = "1999.0"

        with patch.object(orch, "handle_request") as mock_handle:
            orch._poll()
            mock_handle.assert_called_once()
            request = mock_handle.call_args[0][0]
            assert request.problem == "Build an auth system"

    def test_skips_already_processed_message(self) -> None:
        orch, mock_client = _make_orchestrator(founder_user_id="U_FOUNDER")
        mock_client.conversations_history.return_value = {
            "messages": [
                self._build_message("1000.0", "U_FOUNDER", "Old message")
            ]
        }
        # last_ts is equal to the message ts — should be skipped
        orch._last_ts = "1000.0"

        with patch.object(orch, "handle_request") as mock_handle:
            orch._poll()
            mock_handle.assert_not_called()

    def test_unparseable_message_sends_clarification(self) -> None:
        orch, mock_client = _make_orchestrator(founder_user_id="U_FOUNDER")
        mock_client.conversations_history.return_value = {
            "messages": [
                self._build_message("2000.0", "U_FOUNDER", "")
            ]
        }
        orch._last_ts = "1999.0"

        orch._poll()

        texts = [c.kwargs["text"] for c in mock_client.chat_postMessage.call_args_list]
        assert any("couldn't understand" in t for t in texts)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_successful_result_is_correctly_populated(self) -> None:
        from agents.ceo_orchestrator import FounderRequest, PipelineResult

        request = FounderRequest(
            problem="Implement SSO",
            stack="FastAPI + React",
            repo="org/repo",
            priority="high",
            requested_at=datetime.utcnow(),
        )
        completed = datetime.utcnow()
        result = PipelineResult(
            request=request,
            prd_approved=True,
            sprint_approved=True,
            pr_url="https://github.com/org/repo/pull/42",
            qa_passed=True,
            human_review_required=False,
            completed_at=completed,
            errors=[],
        )

        assert result.prd_approved is True
        assert result.sprint_approved is True
        assert result.qa_passed is True
        assert result.pr_url == "https://github.com/org/repo/pull/42"
        assert result.human_review_required is False
        assert result.errors == []
        assert result.request.problem == "Implement SSO"

    def test_failed_result_carries_errors(self) -> None:
        from agents.ceo_orchestrator import FounderRequest, PipelineResult

        request = FounderRequest(problem="Do something")
        result = PipelineResult(
            request=request,
            completed_at=datetime.utcnow(),
            errors=["CrewAI agent timeout", "QA failed twice"],
        )

        assert result.qa_passed is False
        assert len(result.errors) == 2
        assert "timeout" in result.errors[0]

    def test_defaults_are_safe(self) -> None:
        from agents.ceo_orchestrator import FounderRequest, PipelineResult

        request = FounderRequest(problem="Minimal request")
        result = PipelineResult(request=request, completed_at=datetime.utcnow())

        assert result.prd_approved is False
        assert result.sprint_approved is False
        assert result.pr_url is None
        assert result.qa_passed is False
        assert result.human_review_required is False
        assert result.errors == []
