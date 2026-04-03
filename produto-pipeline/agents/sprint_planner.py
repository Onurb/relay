import difflib
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Callable

import anthropic
import requests
from crewai import Agent
from pydantic import BaseModel, Field

from agents.product_thinker import CheckpointError, PRDOutput

logger = logging.getLogger(__name__)

_LLM_MODEL = "claude-opus-4-6"
_LINEAR_API_URL = "https://api.linear.app/graphql"
_MAX_CHECKPOINT_RETRIES = 3
_FIBONACCI_POINTS = {1, 2, 3, 5, 8, 13}
_DUPLICATE_THRESHOLD = 0.8
_CAPACITY_OVERLOAD_THRESHOLD = 1.20   # 20% over capacity
_CAPACITY_UNDERLOAD_THRESHOLD = 0.50  # below 50% of capacity

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DependencyCycleError(Exception):
    """Raised when a cycle is detected in the task dependency graph."""

    def __init__(self, message: str, cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle: list[str] = cycle or []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TechnicalTask(BaseModel):
    """A single technical task derived from a user story."""

    id: str  # format: TASK-001, TASK-002, etc.
    user_story_id: str
    title: str
    description: str
    story_points: int
    suggested_assignee: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    linear_issue_id: str | None = None
    linear_url: str | None = None


class TaskDependency(BaseModel):
    """Explicit dependency record for a task."""

    task_id: str
    depends_on: list[str]


class SprintPlan(BaseModel):
    """The full output of the Sprint Planner agent."""

    sprint_name: str
    total_story_points: int
    capacity_story_points: int
    tasks: list[TechnicalTask]
    ordered_task_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    linear_sprint_id: str | None = None


class PlannerInput(BaseModel):
    """Input consumed by the SprintPlanner agent."""

    prd: PRDOutput
    repo: str


class HistoricalVelocity(BaseModel):
    """Velocity data from a past sprint."""

    sprint_id: str
    planned_points: int
    completed_points: int
    completion_rate: float  # completed / planned


# ---------------------------------------------------------------------------
# SprintPlanner
# ---------------------------------------------------------------------------


class SprintPlanner:
    """Second agent in the pipeline: translates an approved PRD into a sprint.

    Breaks user stories into technical tasks, estimates in Fibonacci points,
    builds a dependency DAG, checks capacity against historical velocity,
    and creates the sprint and issues in Linear after human approval.
    """

    def __init__(
        self,
        llm_client: anthropic.Anthropic,
        linear_token: str,
        github_token: str,
    ) -> None:
        self._llm = llm_client
        self._linear_token = linear_token
        self._github_token = github_token
        self._linear_team_id = os.getenv("LINEAR_TEAM_ID", "")
        self._linear_project_id = os.getenv("LINEAR_PROJECT_ID", "")
        self._sprint_duration_days = int(os.getenv("SPRINT_DURATION_DAYS", "14"))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input: PlannerInput) -> SprintPlan:  # noqa: A002
        """Runs the full sprint planning flow and returns a SprintPlan.

        Does NOT create Linear resources — that happens inside checkpoint()
        after human approval.
        """
        velocity = self._get_velocity()
        open_issues = self._get_open_github_issues(input.repo)

        prompt = self._build_prompt(input, velocity)
        tasks = self._call_llm(prompt)

        duplicate_warnings = self._detect_duplicates(tasks, open_issues)
        capacity_warnings = self._check_capacity(tasks, velocity)
        all_warnings = duplicate_warnings + capacity_warnings

        ordered_ids = self._build_dag(tasks)
        total_points = sum(t.story_points for t in tasks)
        capacity = self._average_velocity(velocity)

        plan = SprintPlan(
            sprint_name=f"Sprint — {input.prd.feature_name}",
            total_story_points=total_points,
            capacity_story_points=capacity,
            tasks=tasks,
            ordered_task_ids=ordered_ids,
            warnings=all_warnings,
        )

        logger.info(
            "SprintPlanner produced %d tasks (%d pts). Warnings: %d",
            len(tasks),
            total_points,
            len(all_warnings),
        )
        return plan

    def checkpoint(
        self,
        plan: SprintPlan,
        notify_fn: Callable[[str], None],
        poll_fn: Callable[[], str | None],
        original_input: PlannerInput,
    ) -> SprintPlan:
        """Sends the sprint plan for human approval, then creates Linear resources.

        Args:
            plan: Initial SprintPlan to submit for review.
            notify_fn: Posts a message to Slack.
            poll_fn: Blocks until the founder replies; returns text or None on timeout.
            original_input: Used for re-runs on rejection.

        Returns:
            Approved SprintPlan with Linear IDs populated.

        Raises:
            CheckpointError: if rejected more than _MAX_CHECKPOINT_RETRIES times.
        """
        current_plan = plan

        for attempt in range(1, _MAX_CHECKPOINT_RETRIES + 1):
            notify_fn(self._format_plan_summary(current_plan, attempt))

            response = poll_fn()
            if response is None:
                raise CheckpointError(
                    "Sprint plan checkpoint timed out — no response from founder."
                )

            normalized = response.strip().lower()

            if normalized == "approve":
                logger.info("Sprint plan approved on attempt %d.", attempt)
                return self._materialise_in_linear(current_plan)

            if normalized.startswith("reject:"):
                feedback = response[len("reject:"):].strip()
                logger.info(
                    "Sprint plan rejected on attempt %d. Feedback: %s",
                    attempt,
                    feedback,
                )
                if attempt < _MAX_CHECKPOINT_RETRIES:
                    velocity = self._get_velocity()
                    prompt = self._build_prompt(original_input, velocity, feedback=feedback)
                    tasks = self._call_llm(prompt)
                    ordered_ids = self._build_dag(tasks)
                    capacity_warnings = self._check_capacity(tasks, velocity)
                    open_issues = self._get_open_github_issues(original_input.repo)
                    dup_warnings = self._detect_duplicates(tasks, open_issues)
                    current_plan = SprintPlan(
                        sprint_name=current_plan.sprint_name,
                        total_story_points=sum(t.story_points for t in tasks),
                        capacity_story_points=current_plan.capacity_story_points,
                        tasks=tasks,
                        ordered_task_ids=ordered_ids,
                        warnings=capacity_warnings + dup_warnings,
                    )
                continue

            notify_fn(
                "🤔 Please reply with *approve* or *reject: <your feedback>* to continue."
            )

        raise CheckpointError(
            f"Sprint plan rejected {_MAX_CHECKPOINT_RETRIES} consecutive times.",
        )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        input: PlannerInput,  # noqa: A002
        velocity: list[HistoricalVelocity],
        feedback: str | None = None,
    ) -> str:
        """Builds the planning prompt from a PlannerInput and historical velocity."""
        avg_velocity = self._average_velocity(velocity)
        velocity_text = (
            "\n".join(
                f"  - {v.sprint_id}: {v.completed_points}/{v.planned_points} pts "
                f"({v.completion_rate:.0%} completion)"
                for v in velocity
            )
            or "  No historical data available."
        )

        stories_text = "\n".join(
            f"  {s.id}: {s.title}\n"
            f"    As a {s.as_a}, I want {s.i_want}, so that {s.so_that}"
            for s in input.prd.user_stories
        )

        feedback_section = (
            f"\n## Previous Rejection Feedback\n{feedback}\n"
            "Revise your output to address this feedback.\n"
            if feedback
            else ""
        )

        schema_example = json.dumps(
            [
                {
                    "id": "TASK-001",
                    "user_story_id": "US-001",
                    "title": "string",
                    "description": "string",
                    "story_points": 3,
                    "suggested_assignee": "backend | frontend | devops | qa | null",
                    "dependencies": [],
                    "linear_issue_id": None,
                    "linear_url": None,
                }
            ],
            indent=2,
        )

        return f"""You are a senior engineering manager planning a two-week sprint.

## Feature Context
Feature: {input.prd.feature_name}
Problem: {input.prd.problem_statement}
Solution: {input.prd.proposed_solution}
Repo: {input.repo}

## User Stories to Break Down
{stories_text}

## Historical Team Velocity (last {len(velocity)} sprints)
{velocity_text}
Average completed points per sprint: {avg_velocity}
{feedback_section}
## Instructions
1. Break each user story into 1-5 concrete technical tasks.
2. Estimate each task in Fibonacci story points: 1, 2, 3, 5, 8, or 13 only.
3. Identify dependencies: if task B cannot start before task A completes,
   add task A's id to task B's dependencies list.
4. Suggest an assignee role based on task type:
   - Backend/API/database work → "backend"
   - UI/frontend work → "frontend"
   - Infrastructure/deployment → "devops"
   - Testing/QA work → "qa"
   - Leave null if unclear
5. Never create tasks that duplicate existing GitHub issues.
6. Keep total story points close to the historical average ({avg_velocity} pts).

## Output Format
Return ONLY a valid JSON array. No markdown, no explanation, just the JSON.
Schema (array of tasks):
{schema_example}
"""

    def _call_llm(self, prompt: str) -> list[TechnicalTask]:
        """Calls the Anthropic API and parses the response into a task list."""
        message = self._llm.messages.create(
            model=_LLM_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> list[TechnicalTask]:
        """Extracts and deserialises the JSON task list from a raw LLM response."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        data: list[dict] = json.loads(text)
        tasks = [TechnicalTask.model_validate(item) for item in data]

        # Clamp story points to nearest Fibonacci value.
        for task in tasks:
            task.story_points = self._nearest_fibonacci(task.story_points)

        return tasks

    # ------------------------------------------------------------------
    # DAG / topological sort
    # ------------------------------------------------------------------

    def _build_dag(self, tasks: list[TechnicalTask]) -> list[str]:
        """Topologically sorts tasks by dependency using Kahn's algorithm.

        Raises:
            DependencyCycleError: if a cycle is detected in the dependency graph.
        """
        task_ids = {t.id for t in tasks}
        # in-degree count and adjacency list
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adj: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    logger.warning(
                        "Task %s depends on unknown task %s — ignoring.", task.id, dep
                    )
                    continue
                adj[dep].append(task.id)
                in_degree[task.id] += 1

        queue: deque[str] = deque(
            tid for tid, degree in in_degree.items() if degree == 0
        )
        ordered: list[str] = []

        while queue:
            tid = queue.popleft()
            ordered.append(tid)
            for neighbour in adj[tid]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(ordered) != len(tasks):
            remaining = [tid for tid in in_degree if in_degree[tid] > 0]
            raise DependencyCycleError(
                f"Dependency cycle detected among tasks: {remaining}",
                cycle=remaining,
            )

        return ordered

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def _check_capacity(
        self,
        tasks: list[TechnicalTask],
        velocity: list[HistoricalVelocity],
    ) -> list[str]:
        """Returns warning strings if planned points deviate significantly from capacity."""
        warnings: list[str] = []
        capacity = self._average_velocity(velocity)
        if capacity == 0:
            return warnings

        total = sum(t.story_points for t in tasks)

        if total > capacity * _CAPACITY_OVERLOAD_THRESHOLD:
            excess = total - capacity
            warnings.append(
                f"Sprint exceeds team capacity by {excess} points "
                f"({total} planned vs {capacity} avg velocity)."
            )

        if total < capacity * _CAPACITY_UNDERLOAD_THRESHOLD:
            shortage = capacity - total
            warnings.append(
                f"Sprint is underloaded by {shortage} points "
                f"({total} planned vs {capacity} avg velocity)."
            )

        return warnings

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _detect_duplicates(
        self,
        tasks: list[TechnicalTask],
        open_issues: list[str],
    ) -> list[str]:
        """Returns warnings for tasks that are similar to open GitHub issues."""
        warnings: list[str] = []
        for task in tasks:
            for issue_title in open_issues:
                ratio = difflib.SequenceMatcher(
                    None, task.title.lower(), issue_title.lower()
                ).ratio()
                if ratio >= _DUPLICATE_THRESHOLD:
                    warnings.append(
                        f"Task '{task.title}' may duplicate open GitHub issue "
                        f"'{issue_title}' (similarity {ratio:.0%})."
                    )
        return warnings

    # ------------------------------------------------------------------
    # Linear integration
    # ------------------------------------------------------------------

    def _get_velocity(self) -> list[HistoricalVelocity]:
        """Queries the last 3 completed cycles from Linear and returns velocity data."""
        query = """
        query GetCycles($teamId: String!) {
          cycles(filter: { team: { id: { eq: $teamId } }, completedAt: { null: false } },
                 orderBy: completedAt, first: 3) {
            nodes {
              id
              name
              issues {
                nodes {
                  estimate
                  completedAt
                }
              }
            }
          }
        }
        """
        try:
            response = self._linear_request(
                query, variables={"teamId": self._linear_team_id}
            )
            cycles = response.get("data", {}).get("cycles", {}).get("nodes", [])
            result: list[HistoricalVelocity] = []
            for cycle in cycles:
                issues = cycle.get("issues", {}).get("nodes", [])
                planned = sum(i.get("estimate") or 0 for i in issues)
                completed = sum(
                    i.get("estimate") or 0
                    for i in issues
                    if i.get("completedAt") is not None
                )
                rate = completed / planned if planned > 0 else 0.0
                result.append(
                    HistoricalVelocity(
                        sprint_id=cycle["id"],
                        planned_points=planned,
                        completed_points=completed,
                        completion_rate=rate,
                    )
                )
            return result
        except Exception as exc:
            logger.warning("Could not fetch Linear velocity: %s", exc)
            return []

    def _create_linear_sprint(self, plan: SprintPlan) -> str:
        """Creates a new Linear cycle and returns its ID."""
        today = datetime.utcnow().date()
        end_date = today + timedelta(days=self._sprint_duration_days)

        mutation = """
        mutation CreateCycle($teamId: String!, $name: String!, $startsAt: DateTime!, $endsAt: DateTime!) {
          cycleCreate(input: {
            teamId: $teamId,
            name: $name,
            startsAt: $startsAt,
            endsAt: $endsAt
          }) {
            cycle { id }
            success
          }
        }
        """
        response = self._linear_request(
            mutation,
            variables={
                "teamId": self._linear_team_id,
                "name": plan.sprint_name,
                "startsAt": today.isoformat() + "T00:00:00Z",
                "endsAt": end_date.isoformat() + "T23:59:59Z",
            },
        )
        cycle_id: str = response["data"]["cycleCreate"]["cycle"]["id"]
        logger.info("Created Linear cycle: %s", cycle_id)
        return cycle_id

    def _create_linear_issues(
        self, tasks: list[TechnicalTask], sprint_id: str
    ) -> list[TechnicalTask]:
        """Creates a Linear issue for each task and returns updated tasks."""
        mutation = """
        mutation CreateIssue($teamId: String!, $title: String!, $description: String!,
                             $estimate: Int, $cycleId: String!) {
          issueCreate(input: {
            teamId: $teamId,
            title: $title,
            description: $description,
            estimate: $estimate,
            cycleId: $cycleId
          }) {
            issue { id url }
            success
          }
        }
        """
        updated: list[TechnicalTask] = []
        for task in tasks:
            try:
                response = self._linear_request(
                    mutation,
                    variables={
                        "teamId": self._linear_team_id,
                        "title": task.title,
                        "description": task.description,
                        "estimate": task.story_points,
                        "cycleId": sprint_id,
                    },
                )
                issue = response["data"]["issueCreate"]["issue"]
                updated.append(
                    task.model_copy(
                        update={
                            "linear_issue_id": issue["id"],
                            "linear_url": issue["url"],
                        }
                    )
                )
                logger.info("Created Linear issue %s for task %s", issue["id"], task.id)
            except Exception as exc:
                logger.error("Failed to create Linear issue for %s: %s", task.id, exc)
                updated.append(task)

        return updated

    def _get_open_github_issues(self, repo: str) -> list[str]:
        """Returns a list of open GitHub issue titles for the given repo."""
        if not repo or "/" not in repo:
            return []
        owner, name = repo.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{name}/issues"
        try:
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._github_token}",
                    "Accept": "application/vnd.github+json",
                },
                params={"state": "open", "per_page": 100},
                timeout=10,
            )
            response.raise_for_status()
            return [issue["title"] for issue in response.json()]
        except Exception as exc:
            logger.warning("Could not fetch GitHub issues for %s: %s", repo, exc)
            return []

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _linear_request(self, query: str, variables: dict | None = None) -> dict:
        """Sends a GraphQL request to the Linear API."""
        response = requests.post(
            _LINEAR_API_URL,
            headers={
                "Authorization": self._linear_token,
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables or {}},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"Linear API error: {data['errors']}")
        return data

    def _materialise_in_linear(self, plan: SprintPlan) -> SprintPlan:
        """Creates the Linear sprint and all issues; returns plan with IDs filled."""
        sprint_id = self._create_linear_sprint(plan)
        updated_tasks = self._create_linear_issues(plan.tasks, sprint_id)
        return plan.model_copy(
            update={
                "linear_sprint_id": sprint_id,
                "tasks": updated_tasks,
            }
        )

    @staticmethod
    def _average_velocity(velocity: list[HistoricalVelocity]) -> int:
        """Returns the rounded average of completed points across past sprints."""
        if not velocity:
            return 0
        return round(sum(v.completed_points for v in velocity) / len(velocity))

    @staticmethod
    def _nearest_fibonacci(points: int) -> int:
        """Clamps a story point value to the nearest Fibonacci number."""
        fib = sorted(_FIBONACCI_POINTS)
        return min(fib, key=lambda f: abs(f - points))

    @staticmethod
    def _format_plan_summary(plan: SprintPlan, attempt: int) -> str:
        """Formats a SprintPlan as a readable Slack message."""
        task_lines = "\n".join(
            f"  • {t.id} ({t.story_points}pts): {t.title}"
            + (f" [{t.suggested_assignee}]" if t.suggested_assignee else "")
            for t in plan.tasks
        )
        order = " → ".join(plan.ordered_task_ids)
        warning_text = (
            "\n⚠️ Warnings:\n" + "\n".join(f"  - {w}" for w in plan.warnings)
            if plan.warnings
            else ""
        )
        return (
            f"🗓️ Sprint Plan ready (attempt {attempt}/{_MAX_CHECKPOINT_RETRIES})\n\n"
            f"*{plan.sprint_name}*\n"
            f"Total: {plan.total_story_points} pts | "
            f"Capacity: {plan.capacity_story_points} pts\n\n"
            f"Tasks ({len(plan.tasks)}):\n{task_lines}\n\n"
            f"Execution order: {order}"
            f"{warning_text}\n\n"
            "Reply *approve* or *reject: <your feedback>*"
        )


# ---------------------------------------------------------------------------
# CrewAI wrapper
# ---------------------------------------------------------------------------


class SprintPlannerAgent:
    """CrewAI Agent wrapper for the SprintPlanner.

    The planning logic lives in SprintPlanner. This class exposes a build()
    method that returns a configured crewai.Agent for use inside ProdutoCrew.
    """

    def build(self) -> Agent:
        """Returns a configured CrewAI Agent for the sprint-planning role."""
        return Agent(
            role="Sprint Planner",
            goal=(
                "Break approved user stories into estimated technical tasks and create "
                "the sprint in Linear with correct dependency ordering."
            ),
            backstory=(
                "You are a senior engineering manager with deep experience in sprint "
                "planning. You always check historical velocity before estimating. You "
                "detect dependencies between tasks and order them correctly. You never "
                "create tasks that duplicate existing GitHub issues."
            ),
            llm=_LLM_MODEL,
            memory=True,
            max_iter=3,
            verbose=True,
        )
