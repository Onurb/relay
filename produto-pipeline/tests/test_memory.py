"""Unit tests for context/memory.py — ProductMemory and its Pydantic models."""

from datetime import datetime
from pathlib import Path

import pytest

from context.memory import (
    ArchitecturalDecision,
    DiscardedFeature,
    ImplementedFeature,
    MemoryContent,
    NamingPattern,
    ProductMemory,
    SprintMemoryUpdate,
    UserFeedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 15, 10, 0, 0)


def _make_memory(tmp_path: Path, file_path: str | None = None) -> ProductMemory:
    path = file_path or str(tmp_path / "memory" / "product_memory.md")
    return ProductMemory(file_path=path)


def _implemented(name: str = "Auth", description: str = "JWT tokens") -> ImplementedFeature:
    return ImplementedFeature(name=name, description=description, date=_NOW)


def _discarded(name: str = "Dark Mode", reason: str = "Low priority") -> DiscardedFeature:
    return DiscardedFeature(name=name, reason=reason, date=_NOW)


# ---------------------------------------------------------------------------
# load() — non-existent file
# ---------------------------------------------------------------------------


class TestLoadNonExistentFile:
    def test_creates_file(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        memory.load()
        assert memory._file_path.exists()

    def test_returns_empty_memory_content(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = memory.load()

        assert isinstance(content, MemoryContent)
        assert content.implemented_features == []
        assert content.architectural_decisions == []
        assert content.discarded_features == []
        assert content.user_feedback == []
        assert content.naming_patterns == []

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested_path = str(tmp_path / "deep" / "nested" / "memory.md")
        memory = ProductMemory(file_path=nested_path)
        memory.load()
        assert Path(nested_path).exists()


# ---------------------------------------------------------------------------
# load() — valid populated file
# ---------------------------------------------------------------------------

_SAMPLE_MEMORY = """---
# Product Memory
Last updated: 2026-01-15T10:00:00

## Implemented Features
- Auth System: JWT-based authentication | PR: https://github.com/org/repo/pull/1 | Date: 2026-01-10T09:00:00
- Dashboard: Analytics overview page | PR: None | Date: 2026-01-12T14:00:00

## Architectural Decisions
- Use PostgreSQL: Relational model fits our data | Date: 2026-01-05T10:00:00

## Discarded Features
- Dark Mode: Low ROI vs effort | Date: 2026-01-08T11:00:00

## User Feedback
- Onboarding: Users find signup confusing | Date: 2026-01-14T09:00:00

## Naming & UX Patterns
- Primary CTA: Always use "Get started" for primary actions
- Error messages: Friendly tone, always suggest next step

---
"""


class TestLoadValidFile:
    def test_parses_implemented_features(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert len(content.implemented_features) == 2
        auth = content.implemented_features[0]
        assert auth.name == "Auth System"
        assert auth.description == "JWT-based authentication"
        assert auth.pr_url == "https://github.com/org/repo/pull/1"
        assert auth.date == datetime(2026, 1, 10, 9, 0, 0)

    def test_pr_none_parsed_as_null(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        dashboard = content.implemented_features[1]
        assert dashboard.pr_url is None

    def test_parses_architectural_decisions(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert len(content.architectural_decisions) == 1
        assert content.architectural_decisions[0].decision == "Use PostgreSQL"
        assert "Relational model" in content.architectural_decisions[0].rationale

    def test_parses_discarded_features(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert len(content.discarded_features) == 1
        assert content.discarded_features[0].name == "Dark Mode"
        assert content.discarded_features[0].reason == "Low ROI vs effort"

    def test_parses_user_feedback(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert len(content.user_feedback) == 1
        assert content.user_feedback[0].theme == "Onboarding"

    def test_parses_naming_patterns(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert len(content.naming_patterns) == 2
        assert content.naming_patterns[0].name == "Primary CTA"

    def test_parses_last_updated(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.md"
        path.write_text(_SAMPLE_MEMORY, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert content.last_updated == datetime(2026, 1, 15, 10, 0, 0)

    def test_empty_sections_parse_to_empty_lists(self, tmp_path: Path) -> None:
        minimal = (
            "---\n# Product Memory\nLast updated: 2026-01-01T00:00:00\n\n"
            "## Implemented Features\n<!-- Add entries after each sprint -->\n\n"
            "## Architectural Decisions\n<!-- Add entries after each sprint -->\n\n"
            "## Discarded Features\n<!-- Add entries after each sprint -->\n\n"
            "## User Feedback\n<!-- Add entries after each sprint -->\n\n"
            "## Naming & UX Patterns\n<!-- Add entries after each sprint -->\n---\n"
        )
        path = tmp_path / "memory.md"
        path.write_text(minimal, encoding="utf-8")
        memory = ProductMemory(file_path=str(path))

        content = memory.load()

        assert content.implemented_features == []
        assert content.discarded_features == []


# ---------------------------------------------------------------------------
# feature_exists()
# ---------------------------------------------------------------------------


class TestFeatureExists:
    def test_returns_true_when_feature_in_implemented(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(
            implemented_features=[_implemented("Auth System")],
        )
        memory._write(content)

        assert memory.feature_exists("Auth System") is True

    def test_case_insensitive(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(implemented_features=[_implemented("Auth System")])
        memory._write(content)

        assert memory.feature_exists("auth system") is True

    def test_returns_false_when_feature_not_present(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(implemented_features=[_implemented("Auth System")])
        memory._write(content)

        assert memory.feature_exists("Dark Mode") is False

    def test_returns_false_on_empty_memory(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        assert memory.feature_exists("Anything") is False


# ---------------------------------------------------------------------------
# was_discarded()
# ---------------------------------------------------------------------------


class TestWasDiscarded:
    def test_returns_true_when_feature_in_discarded(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(discarded_features=[_discarded("Dark Mode")])
        memory._write(content)

        assert memory.was_discarded("Dark Mode") is True

    def test_case_insensitive(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(discarded_features=[_discarded("Dark Mode")])
        memory._write(content)

        assert memory.was_discarded("dark mode") is True

    def test_returns_false_when_not_discarded(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(discarded_features=[_discarded("Dark Mode")])
        memory._write(content)

        assert memory.was_discarded("Auth System") is False

    def test_returns_false_on_empty_memory(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        assert memory.was_discarded("Anything") is False


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_appends_new_features_without_overwriting(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        # Seed with one existing feature.
        content = MemoryContent(implemented_features=[_implemented("Existing Feature")])
        memory._write(content)

        sprint = SprintMemoryUpdate(
            new_features=[_implemented("New Feature", "Brand new")],
        )
        memory.update(sprint)

        reloaded = memory.load()
        names = [f.name for f in reloaded.implemented_features]
        assert "Existing Feature" in names
        assert "New Feature" in names

    def test_appends_decisions(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        sprint = SprintMemoryUpdate(
            new_decisions=[
                ArchitecturalDecision(
                    decision="Use Redis",
                    rationale="Fast cache layer",
                    date=_NOW,
                )
            ]
        )
        memory.update(sprint)

        reloaded = memory.load()
        assert len(reloaded.architectural_decisions) == 1
        assert reloaded.architectural_decisions[0].decision == "Use Redis"

    def test_appends_patterns(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        sprint = SprintMemoryUpdate(
            new_patterns=[NamingPattern(name="Button style", description="Always rounded")]
        )
        memory.update(sprint)

        reloaded = memory.load()
        assert len(reloaded.naming_patterns) == 1
        assert reloaded.naming_patterns[0].name == "Button style"

    def test_updates_last_updated_timestamp(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        original = memory.load()
        original_ts = original.last_updated

        sprint = SprintMemoryUpdate(new_features=[_implemented("New Feature")])
        memory.update(sprint)

        reloaded = memory.load()
        assert reloaded.last_updated >= original_ts

    def test_multiple_updates_accumulate(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)

        memory.update(SprintMemoryUpdate(new_features=[_implemented("Feature A")]))
        memory.update(SprintMemoryUpdate(new_features=[_implemented("Feature B")]))
        memory.update(SprintMemoryUpdate(new_features=[_implemented("Feature C")]))

        reloaded = memory.load()
        assert len(reloaded.implemented_features) == 3

    def test_empty_update_does_not_corrupt_file(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(implemented_features=[_implemented("Existing")])
        memory._write(content)

        memory.update(SprintMemoryUpdate())

        reloaded = memory.load()
        assert len(reloaded.implemented_features) == 1


# ---------------------------------------------------------------------------
# Render / parse round-trip
# ---------------------------------------------------------------------------


class TestRenderParsRoundTrip:
    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        content = MemoryContent(
            implemented_features=[
                ImplementedFeature(
                    name="Auth",
                    description="JWT auth",
                    pr_url="https://github.com/org/repo/pull/5",
                    date=_NOW,
                )
            ],
            architectural_decisions=[
                ArchitecturalDecision(
                    decision="Use Postgres", rationale="ACID compliance", date=_NOW
                )
            ],
            discarded_features=[
                DiscardedFeature(name="Mobile App", reason="Out of scope", date=_NOW)
            ],
            user_feedback=[
                UserFeedback(theme="Speed", summary="Users want faster loads", date=_NOW)
            ],
            naming_patterns=[NamingPattern(name="CTA", description="Use action verbs")],
            last_updated=_NOW,
        )
        memory._write(content)

        reloaded = memory.load()

        assert reloaded.implemented_features[0].name == "Auth"
        assert reloaded.implemented_features[0].pr_url == "https://github.com/org/repo/pull/5"
        assert reloaded.architectural_decisions[0].decision == "Use Postgres"
        assert reloaded.discarded_features[0].name == "Mobile App"
        assert reloaded.user_feedback[0].theme == "Speed"
        assert reloaded.naming_patterns[0].name == "CTA"
