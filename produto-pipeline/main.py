import logging
import os
import sys

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_CHANNEL_ID",
]

AGENTS = [
    "ProductThinkerAgent  — requirements analysis and product brief",
    "SprintPlannerAgent   — sprint decomposition and Linear issues",
    "PromptEngineerAgent  — prompt engineering for code generation",
    "VibeCoderAgent       — code generation and GitHub pull requests",
    "QAAgent              — quality review and PR approval",
]


def validate_env() -> None:
    """Exits with a clear error message if any required env var is missing."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        logger.error(
            "Missing required environment variables: %s", ", ".join(missing)
        )
        print(
            f"ERROR: Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in the values."
        )
        sys.exit(1)


def main() -> None:
    load_dotenv()
    validate_env()

    logger.info("Pipeline v3 started")
    print("Pipeline v3 started\n")
    print("Agents that will be loaded:")
    for agent in AGENTS:
        print(f"  - {agent}")
    print()

    from agents.ceo_orchestrator import CEOOrchestrator

    slack_token = os.environ["SLACK_BOT_TOKEN"]
    slack_channel = os.environ["SLACK_CHANNEL_ID"]

    orchestrator = CEOOrchestrator(
        slack_token=slack_token,
        slack_channel=slack_channel,
    )

    try:
        orchestrator.listen()
    except KeyboardInterrupt:
        logger.info("Shutdown requested — notifying Slack and exiting.")
        orchestrator._notify("🔴 Pipeline going offline.")


if __name__ == "__main__":
    main()
