import logging

from crewai import Agent, Task

from agents.product_thinker import PRDOutput, ProductThinkerAgent, ThinkerInput
from agents.sprint_planner import PlannerInput, SprintPlannerAgent

logger = logging.getLogger(__name__)


def build_product_thinker_agent() -> Agent:
    """Creates and returns the CrewAI Agent for the ProductThinker role."""
    return ProductThinkerAgent().build()


def build_sprint_planner_agent() -> Agent:
    """Creates and returns the CrewAI Agent for the SprintPlanner role."""
    return SprintPlannerAgent().build()


class ProdutoTasks:
    """Defines all tasks for the product pipeline CrewAI Crew.

    Each method returns a configured crewai.Task. product_analysis_task and
    sprint_planning_task are fully implemented; the remaining tasks are
    placeholders to be implemented in subsequent weeks.
    """

    def __init__(
        self,
        thinker_input: ThinkerInput | None = None,
        planner_input: PlannerInput | None = None,
    ) -> None:
        self._thinker_input = thinker_input
        self._planner_input = planner_input

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

    def sprint_planning_task(
        self,
        agent: Agent | None = None,
        planner_input: PlannerInput | None = None,
        context: list[Task] | None = None,
    ) -> Task:
        """CrewAI Task for the SprintPlannerAgent.

        Receives the approved PRD from the ProductThinker and produces a
        structured SprintPlan with Fibonacci-estimated tasks, topological
        ordering, and Linear issue creation after human approval.
        """
        effective_input = planner_input or self._planner_input
        effective_agent = agent or build_sprint_planner_agent()

        description = (
            "Break the approved PRD user stories into technical tasks and plan the sprint.\n\n"
            "Instructions:\n"
            "1. Break each user story into 1-5 concrete technical tasks.\n"
            "2. Estimate each task in Fibonacci story points (1, 2, 3, 5, 8, 13).\n"
            "3. Identify dependencies between tasks and order them correctly.\n"
            "4. Suggest an assignee role per task (backend/frontend/devops/qa).\n"
            "5. Check historical velocity to avoid over- or under-loading the sprint.\n"
            "6. Never create tasks that duplicate open GitHub issues.\n"
        )

        if effective_input:
            description += (
                f"\nFeature: {effective_input.prd.feature_name}\n"
                f"Repo: {effective_input.repo}\n"
                f"User stories: {len(effective_input.prd.user_stories)}\n"
            )

        return Task(
            description=description,
            expected_output=(
                "A structured sprint plan in JSON format matching SprintPlan schema, "
                "with all tasks estimated in Fibonacci points, topologically ordered, "
                "and created in Linear with valid issue IDs and URLs."
            ),
            agent=effective_agent,
            human_input=True,
            context=context or [],
        )

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
