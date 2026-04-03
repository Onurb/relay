import logging
import os
import time
import traceback
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from context.memory import PipelineMemory

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 30
_RETRY_INTERVAL_SECONDS = 60
_QA_MAX_ATTEMPTS = 2


class FounderRequest(BaseModel):
    """Represents a founder's request received via Slack."""

    problem: str
    stack: str = Field(default_factory=lambda: os.getenv("PROJECT_STACK", ""))
    repo: str = Field(default_factory=lambda: os.getenv("GITHUB_REPO", ""))
    priority: str = "normal"
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineResult(BaseModel):
    """Structured output produced after a full pipeline run."""

    request: FounderRequest
    prd_approved: bool = False
    sprint_approved: bool = False
    pr_url: Optional[str] = None
    qa_passed: bool = False
    human_review_required: bool = False
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)


class CEOOrchestrator:
    """Master orchestrator that listens to Slack and drives the product pipeline.

    Polls a configured Slack channel for founder messages, parses them into
    FounderRequest objects, triggers the CrewAI pipeline, and reports back
    at each stage via Slack notifications.
    """

    def __init__(self, slack_token: str, slack_channel: str) -> None:
        self._client = WebClient(token=slack_token)
        self._channel = slack_channel
        self._founder_user_id = os.getenv("SLACK_FOUNDER_USER_ID", "")
        # Initialise to current time so we only pick up messages sent after startup.
        self._last_ts: str = str(time.time())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def listen(self) -> None:
        """Starts a blocking polling loop that processes incoming Slack messages."""
        self._notify("🤖 Product pipeline v3 online. Send me a problem to solve.")
        logger.info("CEO Orchestrator listening on channel %s", self._channel)

        while True:
            try:
                self._poll()
            except SlackApiError as exc:
                logger.error(
                    "Slack unreachable: %s. Retrying in %ds.",
                    exc,
                    _RETRY_INTERVAL_SECONDS,
                )
                time.sleep(_RETRY_INTERVAL_SECONDS)
                continue

            time.sleep(_POLL_INTERVAL_SECONDS)

    def handle_request(self, request: FounderRequest) -> None:
        """Main pipeline trigger: runs the full crew and dispatches Slack updates."""
        self._notify(f"📥 Got it: {request.problem[:100]}... Starting pipeline.")
        logger.info("Handling request: %s", request.problem[:80])

        try:
            result = self._run_pipeline(request)
        except Exception as exc:
            logger.error("Unhandled pipeline error:\n%s", traceback.format_exc())
            self._notify(f"❌ Pipeline error: {exc}. Check logs for details.")
            return

        self._update_memory(result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _notify(self, message: str, level: str = "info") -> None:
        """Posts a message to the Slack channel and logs it locally."""
        try:
            self._client.chat_postMessage(channel=self._channel, text=message)
            logger.log(logging.getLevelName(level.upper()), "Slack → %s", message)
        except SlackApiError as exc:
            logger.error("Failed to post Slack message: %s", exc)

    def _validate_request(self, raw_text: str) -> FounderRequest:
        """Parses a raw Slack message into a FounderRequest.

        Raises:
            ValueError: if raw_text is empty or whitespace-only.
        """
        text = raw_text.strip()
        if not text:
            raise ValueError("Cannot parse an empty message into a FounderRequest.")
        return FounderRequest(problem=text)

    def _run_pipeline(self, request: FounderRequest) -> PipelineResult:
        """Kicks off the CrewAI crew and sends stage notifications to Slack."""
        # Lazy import avoids circular dependency at module load time.
        from crews.produto_crew import ProdutoCrew

        crew = ProdutoCrew()
        result = crew.kickoff(request=request)

        if result.prd_approved:
            self._notify(
                "📋 PRD ready for review. Please approve or reject with feedback."
            )

        if result.sprint_approved:
            self._notify(
                "🗓️ Sprint planned. Please approve or reject with feedback."
            )

        if result.qa_passed and result.pr_url:
            self._notify("✅ PR merged automatically. No action needed.")
        elif not result.qa_passed and result.human_review_required:
            self._notify(
                f"⚠️ QA failed. Human review required: {result.pr_url}"
            )

        return result

    def _update_memory(self, result: PipelineResult) -> None:
        """Persists pipeline result to shared memory after each completed sprint."""
        memory = PipelineMemory()
        memory.store("last_result", result.model_dump(mode="json"))
        logger.info(
            "Pipeline memory updated for request: %s",
            result.request.problem[:60],
        )

    def _poll(self) -> None:
        """Fetches recent channel messages and dispatches any new founder messages."""
        response = self._client.conversations_history(
            channel=self._channel,
            oldest=self._last_ts,
            limit=20,
        )
        messages: list[dict] = response.get("messages", [])

        for msg in reversed(messages):
            ts: str = msg.get("ts", "")
            user: str = msg.get("user", "")
            bot_id: Optional[str] = msg.get("bot_id")
            text: str = msg.get("text", "")

            # Skip bot messages (including our own).
            if bot_id:
                continue

            # Only process messages from the configured founder.
            if self._founder_user_id and user != self._founder_user_id:
                continue

            # Skip messages already processed.
            if ts <= self._last_ts:
                continue

            self._last_ts = ts

            try:
                request = self._validate_request(text)
            except ValueError:
                self._notify(
                    "🤔 I couldn't understand that. "
                    "Please describe the problem or feature you need."
                )
                continue

            self.handle_request(request)
