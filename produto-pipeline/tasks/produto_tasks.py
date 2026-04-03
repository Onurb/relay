import logging

from crewai import Agent, Task

from agents.product_thinker import PRDOutput, ProductThinkerAgent, ThinkerInput

logger = logging.getLogger(__name__)


def build_product_thinker_agent() -> Agent:
    """Creates and returns the CrewAI Agent for the ProductThinker role."""
    return ProductThinkerAgent().build()


class ProdutoTasks:
    """Defines all tasks for the product pipeline CrewAI Crew.

    Each method returns a configured crewai.Task. The product_analysis_task
    is fully implemented; the remaining tasks are placeholders to be
    implemented in subsequent weeks.
    """

    def __init__(self, thinker_input: ThinkerInput | None = None) -> None:
        self._thinker_input = thinker_input

    def product_analysis_task(
        self,
        agent: Agent | None = None,
        thinker_input: ThinkerInput | None = None,
    ) -> Task:
        """CrewAI Task for the ProductThinker agent.

        Produces a structured PRD in JSON format matching PRDOutput, with at
        least 2 user stories and acceptance criteria in Given/When/Then format.
        """
        effective_input = thinker_input or self._thinker_input
        effective_agent = agent or build_product_thinker_agent()

        description = (
            "Analyse the founder's request and produce a structured PRD.\n\n"
            "Instructions:\n"
            "1. Read the product memory context and identify conflicts with existing "
            "or discarded features.\n"
            "2. Identify the real underlying problem behind the request.\n"
            "3. Define minimum viable scope — avoid over-engineering.\n"
            "4. Write at least 2 user stories in As a / I want / So that format.\n"
            "5. Write acceptance criteria in Given / When / Then format.\n"
            "6. Explicitly list what is out of scope.\n"
            "7. Flag open questions that require founder input.\n"
        )

        if effective_input:
            description += (
                f"\nFounder request: {effective_input.request.problem}\n"
                f"Stack: {effective_input.request.stack}\n"
                f"Repo: {effective_input.request.repo}\n"
                f"Priority: {effective_input.request.priority}\n"
            )

        return Task(
            description=description,
            expected_output=(
                "A structured PRD in JSON format matching the PRDOutput schema, "
                "with at least 2 user stories and acceptance criteria in "
                "Given/When/Then format."
            ),
            agent=effective_agent,
            human_input=True,
        )

    def sprint_planning_task(self) -> Task:
        """CrewAI Task for the SprintPlannerAgent.

        TODO: implement in week 2.
        """
        # TODO: implement in week 2 — SprintPlannerAgent
        pass  # type: ignore[return-value]

    def prompt_engineering_task(self) -> Task:
        """CrewAI Task for the PromptEngineerAgent.

        TODO: implement in week 2.
        """
        # TODO: implement in week 2 — PromptEngineerAgent
        pass  # type: ignore[return-value]

    def code_generation_task(self) -> Task:
        """CrewAI Task for the VibeCoderAgent.

        TODO: implement in week 2.
        """
        # TODO: implement in week 2 — VibeCoderAgent
        pass  # type: ignore[return-value]

    def qa_review_task(self) -> Task:
        """CrewAI Task for the QAAgent.

        TODO: implement in week 3.
        """
        # TODO: implement in week 3 — QAAgent
        pass  # type: ignore[return-value]
