"""Unit tests for agents/sprint_planner.py — SprintPlanner and its models."""

import json
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from agents.ceo_orchestrator import FounderRequest
from agents.product_thinker import (
    AcceptanceCriteria,
    CheckpointError,
    PRDOutput,
    UserStory,
)
from agents.sprint_planner import (
    DependencyCycleError,
    HistoricalVelocity,
    PlannerInput,
    SprintPlan,
    SprintPlanner,
    TechnicalTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 15, 10, 0, 0)


def _make_planner() -> tuple[SprintPlanner, MagicMock]:
    mock_client = MagicMock()
    planner = SprintPlanner(
        llm_client=mock_client,
        linear_token="lin_token_fake",
        github_token="ghp_fake",
    )
    return planner, mock_client


def _make_task(
    id_: str = "TASK-001",
    title: str = "Set up database",
    story_points: int = 3,
    dependencies: list[str] | None = None,
    user_story_id: str = "US-001",
) -> TechnicalTask:
    return TechnicalTask(
        id=id_,
        user_story_id=user_story_id,
        title=title,
        description=f"Description for {id_}",
        story_points=story_points,
        dependencies=dependencies or [],
    )


def _make_velocity(
    sprint_id: str = "CYC-1",
    planned: int = 20,
    completed: int = 18,
) -> HistoricalVelocity:
    return HistoricalVelocity(
        sprint_id=sprint_id,
        planned_points=planned,
        completed_points=completed,
        completion_rate=completed / planned,
    )


def _make_prd() -> PRDOutput:
    return PRDOutput(
        feature_name="Auth System",
        problem_statement="Users cannot log in",
        proposed_solution="JWT auth",
        out_of_scope=["SSO"],
        user_stories=[
            UserStory(
                id="US-001",
                title="Login",
                as_a="user",
                i_want="to log in",
                so_that="I can access my account",
                acceptance_criteria=[
                    AcceptanceCriteria(given="on login page", when="submit creds", then="redirected")
                ],
            )
        ],
        open_questions=[],
        created_at=_NOW,
    )


def _make_plan(
    tasks: list[TechnicalTask] | None = None,
    warnings: list[str] | None = None,
) -> SprintPlan:
    t = tasks or [_make_task()]
    return SprintPlan(
        sprint_name="Sprint — Auth System",
        total_story_points=sum(tk.story_points for tk in t),
        capacity_story_points=20,
        tasks=t,
        ordered_task_ids=[tk.id for tk in t],
        warnings=warnings or [],
        created_at=_NOW,
    )


def _stub_llm(mock_client: MagicMock, tasks: list[TechnicalTask]) -> None:
    raw = json.dumps([t.model_dump() for t in tasks])
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=raw)]
    mock_client.messages.create.return_value = mock_message


# ---------------------------------------------------------------------------
# _build_dag() — valid graph
# ---------------------------------------------------------------------------


class TestBuildDag:
    def test_tasks_with_no_dependencies_come_first(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001", dependencies=[]),
            _make_task("TASK-002", dependencies=["TASK-001"]),
            _make_task("TASK-003", dependencies=["TASK-002"]),
        ]
        order = planner._build_dag(tasks)
        assert order.index("TASK-001") < order.index("TASK-002")
        assert order.index("TASK-002") < order.index("TASK-003")

    def test_independent_tasks_all_included(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001"),
            _make_task("TASK-002"),
            _make_task("TASK-003"),
        ]
        order = planner._build_dag(tasks)
        assert set(order) == {"TASK-001", "TASK-002", "TASK-003"}

    def test_diamond_dependency_correct_order(self) -> None:
        planner, _ = _make_planner()
        #   001
        #  /   \
        # 002  003
        #  \   /
        #   004
        tasks = [
            _make_task("TASK-001", dependencies=[]),
            _make_task("TASK-002", dependencies=["TASK-001"]),
            _make_task("TASK-003", dependencies=["TASK-001"]),
            _make_task("TASK-004", dependencies=["TASK-002", "TASK-003"]),
        ]
        order = planner._build_dag(tasks)
        assert order[0] == "TASK-001"
        assert order[-1] == "TASK-004"
        assert order.index("TASK-002") < order.index("TASK-004")
        assert order.index("TASK-003") < order.index("TASK-004")

    def test_single_task_no_deps(self) -> None:
        planner, _ = _make_planner()
        order = planner._build_dag([_make_task("TASK-001")])
        assert order == ["TASK-001"]

    def test_unknown_dependency_is_ignored(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001", dependencies=["TASK-UNKNOWN"])]
        order = planner._build_dag(tasks)
        assert "TASK-001" in order


# ---------------------------------------------------------------------------
# _build_dag() — cycle detection
# ---------------------------------------------------------------------------


class TestBuildDagCycles:
    def test_simple_cycle_raises_dependency_cycle_error(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001", dependencies=["TASK-002"]),
            _make_task("TASK-002", dependencies=["TASK-001"]),
        ]
        with pytest.raises(DependencyCycleError) as exc_info:
            planner._build_dag(tasks)
        assert exc_info.value.cycle

    def test_three_node_cycle_raises(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001", dependencies=["TASK-003"]),
            _make_task("TASK-002", dependencies=["TASK-001"]),
            _make_task("TASK-003", dependencies=["TASK-002"]),
        ]
        with pytest.raises(DependencyCycleError):
            planner._build_dag(tasks)

    def test_cycle_error_contains_affected_task_ids(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-A", dependencies=["TASK-B"]),
            _make_task("TASK-B", dependencies=["TASK-A"]),
        ]
        with pytest.raises(DependencyCycleError) as exc_info:
            planner._build_dag(tasks)
        cycle = exc_info.value.cycle
        assert "TASK-A" in cycle or "TASK-B" in cycle


# ---------------------------------------------------------------------------
# _check_capacity()
# ---------------------------------------------------------------------------


class TestCheckCapacity:
    def test_sprint_exceeding_capacity_by_25_percent_warns(self) -> None:
        planner, _ = _make_planner()
        # avg velocity = 20, sprint = 25 pts → 25% over
        velocity = [_make_velocity(completed=20)] * 3
        tasks = [_make_task(story_points=5)] * 5  # 25 pts

        warnings = planner._check_capacity(tasks, velocity)

        assert len(warnings) == 1
        assert "exceeds" in warnings[0].lower()

    def test_sprint_at_40_percent_capacity_warns_underloaded(self) -> None:
        planner, _ = _make_planner()
        # avg velocity = 20, sprint = 8 pts → 40% of capacity
        velocity = [_make_velocity(completed=20)] * 3
        tasks = [_make_task(story_points=8)]

        warnings = planner._check_capacity(tasks, velocity)

        assert len(warnings) == 1
        assert "underloaded" in warnings[0].lower()

    def test_sprint_at_90_percent_capacity_no_warnings(self) -> None:
        planner, _ = _make_planner()
        # avg velocity = 20, sprint = 18 pts → 90%
        velocity = [_make_velocity(completed=20)] * 3
        tasks = [
            _make_task("TASK-001", story_points=8),
            _make_task("TASK-002", story_points=5),
            _make_task("TASK-003", story_points=5),
        ]  # 18 pts

        warnings = planner._check_capacity(tasks, velocity)

        assert warnings == []

    def test_no_velocity_data_returns_no_warnings(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task(story_points=5)] * 3

        warnings = planner._check_capacity(tasks, [])

        assert warnings == []

    def test_exact_capacity_no_warnings(self) -> None:
        planner, _ = _make_planner()
        velocity = [_make_velocity(completed=15)] * 3  # avg = 15
        tasks = [_make_task(story_points=5)] * 3  # exactly 15 pts

        warnings = planner._check_capacity(tasks, velocity)

        assert warnings == []


# ---------------------------------------------------------------------------
# _detect_duplicates()
# ---------------------------------------------------------------------------


class TestDetectDuplicates:
    def test_similar_title_returns_warning(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001", title="Set up authentication database")]
        open_issues = ["Set up authentication database schema"]

        warnings = planner._detect_duplicates(tasks, open_issues)

        assert len(warnings) == 1
        assert "duplicate" in warnings[0].lower()

    def test_different_titles_returns_empty(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001", title="Build login form")]
        open_issues = ["Fix broken deployment pipeline", "Update README docs"]

        warnings = planner._detect_duplicates(tasks, open_issues)

        assert warnings == []

    def test_exact_match_returns_warning(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001", title="Add JWT middleware")]
        open_issues = ["Add JWT middleware"]

        warnings = planner._detect_duplicates(tasks, open_issues)

        assert len(warnings) == 1

    def test_empty_open_issues_returns_empty(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001", title="Anything")]

        warnings = planner._detect_duplicates(tasks, [])

        assert warnings == []

    def test_multiple_tasks_partial_duplicates(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001", title="Create user migration"),
            _make_task("TASK-002", title="Build payment gateway"),
        ]
        open_issues = ["Create user database migration"]

        warnings = planner._detect_duplicates(tasks, open_issues)

        # Only TASK-001 should match
        assert len(warnings) == 1
        assert "Create user migration" in warnings[0]


# ---------------------------------------------------------------------------
# checkpoint() — approve path
# ---------------------------------------------------------------------------


class TestCheckpointApprove:
    def test_approve_calls_create_linear_sprint(self) -> None:
        planner, _ = _make_planner()
        plan = _make_plan()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="approve")

        with patch.object(planner, "_materialise_in_linear", return_value=plan) as mock_mat:
            planner.checkpoint(plan, notify_fn, poll_fn, MagicMock())
            mock_mat.assert_called_once_with(plan)

    def test_approve_returns_plan_with_linear_ids(self) -> None:
        planner, _ = _make_planner()
        plan = _make_plan()
        plan_with_ids = plan.model_copy(update={"linear_sprint_id": "CYC-123"})
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="approve")

        with patch.object(planner, "_materialise_in_linear", return_value=plan_with_ids):
            result = planner.checkpoint(plan, notify_fn, poll_fn, MagicMock())

        assert result.linear_sprint_id == "CYC-123"

    def test_approve_sends_formatted_summary_to_slack(self) -> None:
        planner, _ = _make_planner()
        plan = _make_plan()
        notify_fn = MagicMock()
        poll_fn = MagicMock(return_value="approve")

        with patch.object(planner, "_materialise_in_linear", return_value=plan):
            planner.checkpoint(plan, notify_fn, poll_fn, MagicMock())

        notify_fn.assert_called_once()
        msg = notify_fn.call_args[0][0]
        assert "Sprint Plan ready" in msg
        assert str(plan.total_story_points) in msg


# ---------------------------------------------------------------------------
# checkpoint() — reject path
# ---------------------------------------------------------------------------


class TestCheckpointReject:
    def test_reject_reruns_with_feedback_in_prompt(self) -> None:
        planner, mock_client = _make_planner()
        original_plan = _make_plan()
        revised_tasks = [_make_task("TASK-001", story_points=2), _make_task("TASK-002", story_points=3)]
        _stub_llm(mock_client, revised_tasks)

        notify_fn = MagicMock()
        poll_fn = MagicMock(side_effect=["reject: split task 3", "approve"])
        original_input = PlannerInput(prd=_make_prd(), repo="org/repo")

        with patch.object(planner, "_get_velocity", return_value=[_make_velocity()]), \
             patch.object(planner, "_get_open_github_issues", return_value=[]), \
             patch.object(planner, "_materialise_in_linear", side_effect=lambda p: p):
            planner.checkpoint(original_plan, notify_fn, poll_fn, original_input)

        # LLM called once for the re-run
        assert mock_client.messages.create.call_count == 1
        prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        assert "split task 3" in prompt

    def test_reject_does_not_create_linear_until_approved(self) -> None:
        planner, mock_client = _make_planner()
        plan = _make_plan()
        _stub_llm(mock_client, plan.tasks)

        notify_fn = MagicMock()
        poll_fn = MagicMock(side_effect=["reject: not good", "approve"])
        original_input = PlannerInput(prd=_make_prd(), repo="org/repo")

        with patch.object(planner, "_get_velocity", return_value=[]), \
             patch.object(planner, "_get_open_github_issues", return_value=[]), \
             patch.object(planner, "_materialise_in_linear", side_effect=lambda p: p) as mock_mat:
            planner.checkpoint(plan, notify_fn, poll_fn, original_input)

        # _materialise_in_linear called exactly once (on approve)
        mock_mat.assert_called_once()


# ---------------------------------------------------------------------------
# checkpoint() — max retries exhausted
# ---------------------------------------------------------------------------


class TestCheckpointMaxRetries:
    def test_raises_checkpoint_error_after_3_rejections(self) -> None:
        planner, mock_client = _make_planner()
        plan = _make_plan()
        _stub_llm(mock_client, plan.tasks)

        notify_fn = MagicMock()
        poll_fn = MagicMock(
            side_effect=["reject: 1", "reject: 2", "reject: 3"]
        )
        original_input = PlannerInput(prd=_make_prd(), repo="org/repo")

        with patch.object(planner, "_get_velocity", return_value=[]), \
             patch.object(planner, "_get_open_github_issues", return_value=[]):
            with pytest.raises(CheckpointError):
                planner.checkpoint(plan, notify_fn, poll_fn, original_input)


# ---------------------------------------------------------------------------
# _get_velocity() — mocked Linear response
# ---------------------------------------------------------------------------


class TestGetVelocity:
    def test_returns_correct_velocity_list(self) -> None:
        planner, _ = _make_planner()
        mock_response = {
            "data": {
                "cycles": {
                    "nodes": [
                        {
                            "id": "CYC-1",
                            "name": "Sprint 1",
                            "issues": {
                                "nodes": [
                                    {"estimate": 5, "completedAt": "2026-01-10T00:00:00Z"},
                                    {"estimate": 3, "completedAt": "2026-01-10T00:00:00Z"},
                                    {"estimate": 8, "completedAt": None},
                                ]
                            },
                        },
                        {
                            "id": "CYC-2",
                            "name": "Sprint 2",
                            "issues": {
                                "nodes": [
                                    {"estimate": 13, "completedAt": "2026-01-24T00:00:00Z"},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        with patch.object(planner, "_linear_request", return_value=mock_response):
            result = planner._get_velocity()

        assert len(result) == 2
        sprint1 = result[0]
        assert sprint1.sprint_id == "CYC-1"
        assert sprint1.planned_points == 16   # 5+3+8
        assert sprint1.completed_points == 8  # 5+3 (completedAt not None)
        assert abs(sprint1.completion_rate - 0.5) < 0.01

        sprint2 = result[1]
        assert sprint2.planned_points == 13
        assert sprint2.completed_points == 13
        assert sprint2.completion_rate == 1.0

    def test_returns_empty_list_on_linear_error(self) -> None:
        planner, _ = _make_planner()

        with patch.object(planner, "_linear_request", side_effect=Exception("timeout")):
            result = planner._get_velocity()

        assert result == []


# ---------------------------------------------------------------------------
# _create_linear_issues() — mocked Linear API
# ---------------------------------------------------------------------------


class TestCreateLinearIssues:
    def test_returns_tasks_with_linear_ids_filled(self) -> None:
        planner, _ = _make_planner()
        tasks = [
            _make_task("TASK-001", title="Create schema"),
            _make_task("TASK-002", title="Add endpoints"),
        ]

        def mock_linear_request(query: str, variables: dict | None = None) -> dict:
            title = (variables or {}).get("title", "unknown")
            return {
                "data": {
                    "issueCreate": {
                        "issue": {
                            "id": f"ISSUE-{title[:3].upper()}",
                            "url": f"https://linear.app/issue/{title[:3].lower()}",
                        },
                        "success": True,
                    }
                }
            }

        with patch.object(planner, "_linear_request", side_effect=mock_linear_request):
            result = planner._create_linear_issues(tasks, "CYC-999")

        assert result[0].linear_issue_id is not None
        assert result[0].linear_url is not None
        assert result[1].linear_issue_id is not None

    def test_failed_issue_creation_keeps_task_without_id(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001")]

        with patch.object(
            planner, "_linear_request", side_effect=Exception("Linear API down")
        ):
            result = planner._create_linear_issues(tasks, "CYC-999")

        # Task is kept but without linear_issue_id
        assert result[0].linear_issue_id is None

    def test_all_tasks_returned_even_on_partial_failure(self) -> None:
        planner, _ = _make_planner()
        tasks = [_make_task("TASK-001"), _make_task("TASK-002")]
        call_count = 0

        def flaky_request(query: str, variables: dict | None = None) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("temporary error")
            return {
                "data": {
                    "issueCreate": {
                        "issue": {"id": "ISSUE-OK", "url": "https://linear.app/ok"},
                        "success": True,
                    }
                }
            }

        with patch.object(planner, "_linear_request", side_effect=flaky_request):
            result = planner._create_linear_issues(tasks, "CYC-999")

        assert len(result) == 2
        assert result[0].linear_issue_id is None   # failed
        assert result[1].linear_issue_id == "ISSUE-OK"  # succeeded


# ---------------------------------------------------------------------------
# _parse_response()
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _raw_tasks_json(self, tasks: list[TechnicalTask] | None = None) -> str:
        t = tasks or [_make_task()]
        return json.dumps([task.model_dump() for task in t])

    def test_parses_plain_json_array(self) -> None:
        planner, _ = _make_planner()
        result = planner._parse_response(self._raw_tasks_json())
        assert len(result) == 1
        assert isinstance(result[0], TechnicalTask)

    def test_parses_json_in_code_fence(self) -> None:
        planner, _ = _make_planner()
        wrapped = f"```json\n{self._raw_tasks_json()}\n```"
        result = planner._parse_response(wrapped)
        assert isinstance(result[0], TechnicalTask)

    def test_story_points_clamped_to_fibonacci(self) -> None:
        planner, _ = _make_planner()
        task = _make_task(story_points=4)  # 4 is not Fibonacci → nearest is 3 or 5
        result = planner._parse_response(self._raw_tasks_json([task]))
        assert result[0].story_points in {3, 5}

    def test_story_points_7_clamped_to_8(self) -> None:
        planner, _ = _make_planner()
        task = _make_task(story_points=7)
        result = planner._parse_response(self._raw_tasks_json([task]))
        assert result[0].story_points == 8
