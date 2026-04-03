import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ImplementedFeature(BaseModel):
    """A feature that has been shipped to production."""

    name: str
    description: str
    pr_url: str | None = None
    date: datetime


class ArchitecturalDecision(BaseModel):
    """A recorded architectural decision with its rationale."""

    decision: str
    rationale: str
    date: datetime


class DiscardedFeature(BaseModel):
    """A feature that was explicitly decided against."""

    name: str
    reason: str
    date: datetime


class UserFeedback(BaseModel):
    """A theme of user feedback collected over time."""

    theme: str
    summary: str
    date: datetime


class NamingPattern(BaseModel):
    """A naming or UX convention followed in the product."""

    name: str
    description: str


class MemoryContent(BaseModel):
    """Full contents of the product memory file."""

    implemented_features: list[ImplementedFeature] = Field(default_factory=list)
    architectural_decisions: list[ArchitecturalDecision] = Field(default_factory=list)
    discarded_features: list[DiscardedFeature] = Field(default_factory=list)
    user_feedback: list[UserFeedback] = Field(default_factory=list)
    naming_patterns: list[NamingPattern] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class SprintMemoryUpdate(BaseModel):
    """Data produced at the end of a sprint to be appended to memory."""

    new_features: list[ImplementedFeature] = Field(default_factory=list)
    new_decisions: list[ArchitecturalDecision] = Field(default_factory=list)
    new_patterns: list[NamingPattern] = Field(default_factory=list)
    problems_encountered: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ProductMemory
# ---------------------------------------------------------------------------

_SECTION_IMPLEMENTED = "Implemented Features"
_SECTION_DECISIONS = "Architectural Decisions"
_SECTION_DISCARDED = "Discarded Features"
_SECTION_FEEDBACK = "User Feedback"
_SECTION_PATTERNS = "Naming & UX Patterns"


class ProductMemory:
    """Manages a persistent markdown file that stores cross-sprint product knowledge.

    The file follows a fixed section-based format. Entries are only ever
    appended — existing content is never overwritten.
    """

    def __init__(
        self,
        source: str = "file",
        file_path: str = "memory/product_memory.md",
    ) -> None:
        self._source = source
        self._file_path = Path(file_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> MemoryContent:
        """Reads and parses the memory file.

        If the file does not exist it is created with empty sections and
        an empty MemoryContent is returned.
        """
        if not self._file_path.exists():
            logger.info("Memory file not found — creating at %s", self._file_path)
            empty = MemoryContent()
            self._write(empty)
            return empty

        raw = self._file_path.read_text(encoding="utf-8")
        return self._parse(raw)

    def update(self, sprint_result: SprintMemoryUpdate) -> None:
        """Appends new entries from a sprint result without overwriting anything."""
        content = self.load()

        content.implemented_features.extend(sprint_result.new_features)
        content.architectural_decisions.extend(sprint_result.new_decisions)
        content.naming_patterns.extend(sprint_result.new_patterns)
        content.last_updated = datetime.utcnow()

        self._write(content)
        logger.info(
            "Memory updated: +%d features, +%d decisions, +%d patterns",
            len(sprint_result.new_features),
            len(sprint_result.new_decisions),
            len(sprint_result.new_patterns),
        )

    def feature_exists(self, feature_name: str) -> bool:
        """Returns True if a feature with this name is already implemented."""
        content = self.load()
        needle = feature_name.lower()
        return any(f.name.lower() == needle for f in content.implemented_features)

    def was_discarded(self, feature_name: str) -> bool:
        """Returns True if this feature was explicitly discarded."""
        content = self.load()
        needle = feature_name.lower()
        return any(f.name.lower() == needle for f in content.discarded_features)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write(self, content: MemoryContent) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(self._render(content), encoding="utf-8")

    def _parse(self, raw: str) -> MemoryContent:
        """Parses the markdown memory file into a MemoryContent object."""
        sections: dict[str, list[str]] = {}
        current_section: str | None = None
        last_updated = datetime.utcnow()

        for line in raw.splitlines():
            stripped = line.strip()

            if stripped.startswith("Last updated:"):
                try:
                    last_updated = datetime.fromisoformat(
                        stripped.replace("Last updated:", "").strip()
                    )
                except ValueError:
                    pass
                continue

            if stripped.startswith("## "):
                current_section = stripped[3:].strip()
                sections.setdefault(current_section, [])
                continue

            if current_section and stripped.startswith("- "):
                sections[current_section].append(stripped[2:])

        return MemoryContent(
            implemented_features=self._parse_implemented(
                sections.get(_SECTION_IMPLEMENTED, [])
            ),
            architectural_decisions=self._parse_decisions(
                sections.get(_SECTION_DECISIONS, [])
            ),
            discarded_features=self._parse_discarded(
                sections.get(_SECTION_DISCARDED, [])
            ),
            user_feedback=self._parse_feedback(
                sections.get(_SECTION_FEEDBACK, [])
            ),
            naming_patterns=self._parse_patterns(
                sections.get(_SECTION_PATTERNS, [])
            ),
            last_updated=last_updated,
        )

    def _render(self, content: MemoryContent) -> str:
        """Renders a MemoryContent object back to the canonical markdown format."""
        lines: list[str] = [
            "---",
            "# Product Memory",
            f"Last updated: {content.last_updated.isoformat()}",
            "",
        ]

        lines += [f"## {_SECTION_IMPLEMENTED}"]
        if content.implemented_features:
            for f in content.implemented_features:
                pr = f.pr_url or "None"
                lines.append(
                    f"- {f.name}: {f.description} | PR: {pr} | Date: {f.date.isoformat()}"
                )
        else:
            lines.append("<!-- Add entries after each sprint -->")
        lines.append("")

        lines += [f"## {_SECTION_DECISIONS}"]
        if content.architectural_decisions:
            for d in content.architectural_decisions:
                lines.append(
                    f"- {d.decision}: {d.rationale} | Date: {d.date.isoformat()}"
                )
        else:
            lines.append("<!-- Add entries after each sprint -->")
        lines.append("")

        lines += [f"## {_SECTION_DISCARDED}"]
        if content.discarded_features:
            for f in content.discarded_features:
                lines.append(
                    f"- {f.name}: {f.reason} | Date: {f.date.isoformat()}"
                )
        else:
            lines.append("<!-- Add entries after each sprint -->")
        lines.append("")

        lines += [f"## {_SECTION_FEEDBACK}"]
        if content.user_feedback:
            for fb in content.user_feedback:
                lines.append(
                    f"- {fb.theme}: {fb.summary} | Date: {fb.date.isoformat()}"
                )
        else:
            lines.append("<!-- Add entries after each sprint -->")
        lines.append("")

        lines += [f"## {_SECTION_PATTERNS}"]
        if content.naming_patterns:
            for p in content.naming_patterns:
                lines.append(f"- {p.name}: {p.description}")
        else:
            lines.append("<!-- Add entries after each sprint -->")
        lines.append("")

        lines.append("---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(raw: str) -> datetime:
        try:
            return datetime.fromisoformat(raw.strip())
        except ValueError:
            return datetime.utcnow()

    @staticmethod
    def _split_pipe(entry: str, expected: int) -> list[str] | None:
        parts = [p.strip() for p in entry.split("|")]
        return parts if len(parts) >= expected else None

    def _parse_implemented(self, entries: list[str]) -> list[ImplementedFeature]:
        result = []
        for entry in entries:
            parts = self._split_pipe(entry, 3)
            if not parts:
                continue
            name_desc = parts[0].split(":", 1)
            if len(name_desc) != 2:
                continue
            name, description = name_desc[0].strip(), name_desc[1].strip()
            pr_raw = parts[1].replace("PR:", "").strip()
            pr_url = pr_raw if pr_raw.lower() not in ("none", "") else None
            date = self._parse_date(parts[2].replace("Date:", ""))
            result.append(
                ImplementedFeature(
                    name=name, description=description, pr_url=pr_url, date=date
                )
            )
        return result

    def _parse_decisions(self, entries: list[str]) -> list[ArchitecturalDecision]:
        result = []
        for entry in entries:
            parts = self._split_pipe(entry, 2)
            if not parts:
                continue
            dec_rat = parts[0].split(":", 1)
            if len(dec_rat) != 2:
                continue
            decision, rationale = dec_rat[0].strip(), dec_rat[1].strip()
            date = self._parse_date(parts[1].replace("Date:", ""))
            result.append(
                ArchitecturalDecision(decision=decision, rationale=rationale, date=date)
            )
        return result

    def _parse_discarded(self, entries: list[str]) -> list[DiscardedFeature]:
        result = []
        for entry in entries:
            parts = self._split_pipe(entry, 2)
            if not parts:
                continue
            name_reason = parts[0].split(":", 1)
            if len(name_reason) != 2:
                continue
            name, reason = name_reason[0].strip(), name_reason[1].strip()
            date = self._parse_date(parts[1].replace("Date:", ""))
            result.append(DiscardedFeature(name=name, reason=reason, date=date))
        return result

    def _parse_feedback(self, entries: list[str]) -> list[UserFeedback]:
        result = []
        for entry in entries:
            parts = self._split_pipe(entry, 2)
            if not parts:
                continue
            theme_summary = parts[0].split(":", 1)
            if len(theme_summary) != 2:
                continue
            theme, summary = theme_summary[0].strip(), theme_summary[1].strip()
            date = self._parse_date(parts[1].replace("Date:", ""))
            result.append(UserFeedback(theme=theme, summary=summary, date=date))
        return result

    def _parse_patterns(self, entries: list[str]) -> list[NamingPattern]:
        result = []
        for entry in entries:
            name_desc = entry.split(":", 1)
            if len(name_desc) != 2:
                continue
            result.append(
                NamingPattern(
                    name=name_desc[0].strip(), description=name_desc[1].strip()
                )
            )
        return result


# ---------------------------------------------------------------------------
# PipelineMemory (session-scoped in-memory store, used by CEOOrchestrator)
# ---------------------------------------------------------------------------


class PipelineMemory:
    """Lightweight in-memory key-value store shared across agents within a session."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Stores a value under the given key."""
        self._store[key] = value

    def retrieve(self, key: str) -> Any:
        """Returns the value for a key, or None if not found."""
        return self._store.get(key)

    def clear(self) -> None:
        """Clears all stored values."""
        self._store.clear()
