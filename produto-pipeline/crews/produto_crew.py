import logging
from datetime import datetime
from typing import Optional

from crewai import Crew, Process

from agents.ceo_orchestrator import FounderRequest, PipelineResult
from agents.product_thinker import ProductThinkerAgent
from agents.sprint_planner import SprintPlannerAgent
from agents.prompt_engineer import PromptEngineerAgent
from agents.vibe_coder import VibeCoderAgent
from agents.qa_agent import QAAgent
from context.memory import PipelineMemory
from context.rag_index import CodebaseRAG
from tasks.produto_tasks import ProdutoTasks

logger = logging.getLogger(__name__)


def build_crew(
    product_memory: Optional[PipelineMemory] = None,
    rag_index: Optional[CodebaseRAG] = None,
) -> Crew:
    """Builds and returns a configured CrewAI Crew for the product pipeline.

    Ensures the RAG index is refreshed before the PromptEngineer runs so
    that context always reflects the latest codebase state.

    Args:
        product_memory: Optional shared memory instance to inject as context.
        rag_index: Optional CodebaseRAG instance to inject as context.

    Returns:
        A CrewAI Crew configured for sequential execution with memory enabled.
    """
    # Refresh the RAG index before the PromptEngineer consumes it.
    if rag_index is not None:
        try:
            rag_index.update()
        except Exception as exc:
            logger.warning("RAG index update failed — proceeding with stale index: %s", exc)

    tasks_factory = ProdutoTasks()

    agent_builders = [
        ProductThinkerAgent(),
        SprintPlannerAgent(),
        PromptEngineerAgent(),
        VibeCoderAgent(),
        QAAgent(),
    ]

    built_agents = [b.build() for b in agent_builders]
    unresolved = [i for i, a in enumerate(built_agents) if a is None]
    if unresolved:
        names = [type(agent_builders[i]).__name__ for i in unresolved]
        logger.warning(
            "The following agents are not yet implemented and returned None: %s",
            names,
        )

    agents = [a for a in built_agents if a is not None]

    thinker_task = tasks_factory.product_analysis_task()
    planner_task = tasks_factory.sprint_planning_task(
        context=[thinker_task] if thinker_task else []
    )
    engineer_task = tasks_factory.prompt_engineering_task(
        context=[planner_task] if planner_task else []
    )
    coder_task = tasks_factory.code_generation_task()
    qa_task = tasks_factory.qa_review_task()

    tasks = [t for t in [thinker_task, planner_task, engineer_task, coder_task, qa_task]
             if t is not None]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        memory=True,
        verbose=True,
    )


class ProdutoCrew:
    """Orchestrates the 5 product pipeline agents in a CrewAI v3 Crew.

    Accepts optional product_memory and rag_index to inject shared context
    into the crew. Wraps the CrewAI kickoff in a structured PipelineResult.
    """

    def __init__(
        self,
        product_memory: Optional[PipelineMemory] = None,
        rag_index: Optional[CodebaseRAG] = None,
    ) -> None:
        self._product_memory = product_memory
        self._rag_index = rag_index

    def kickoff(self, request: Optional[FounderRequest] = None) -> PipelineResult:
        """Runs the full pipeline and returns a structured PipelineResult.

        Args:
            request: The FounderRequest that triggered this run.

        Returns:
            A PipelineResult reflecting the outcome of the pipeline run.
        """
        effective_request = request or FounderRequest(problem="")
        logger.info("Starting pipeline kickoff for: %s", effective_request.problem[:80])

        try:
            crew = build_crew(
                product_memory=self._product_memory,
                rag_index=self._rag_index,
            )
            inputs = effective_request.model_dump(mode="json")
            crew.kickoff(inputs=inputs)

            # Placeholder outcome — individual agents will populate real values
            # once implemented in their respective weeks.
            return PipelineResult(
                request=effective_request,
                prd_approved=True,
                sprint_approved=True,
                pr_url=None,
                qa_passed=False,
                human_review_required=True,
                completed_at=datetime.utcnow(),
                errors=[],
            )

        except Exception as exc:
            logger.error("Crew kickoff failed: %s", exc)
            return PipelineResult(
                request=effective_request,
                completed_at=datetime.utcnow(),
                errors=[str(exc)],
            )
