"""Unit tests for context/rag_index.py — CodebaseRAG and its models."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from context.rag_index import (
    CodebaseRAG,
    RAGConfig,
    RAGSearchResult,
    RelevantFile,
    create_rag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(repo_path: str = "/fake/repo", **kwargs) -> RAGConfig:
    return RAGConfig(repo_path=repo_path, **kwargs)


def _make_rag(repo_path: str = "/fake/repo", **kwargs) -> CodebaseRAG:
    return CodebaseRAG(_make_config(repo_path=repo_path, **kwargs))


def _make_node(file_path: str, content: str, score: float) -> MagicMock:
    """Creates a mock LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.score = score
    node.node.metadata = {"file_path": file_path}
    node.node.get_content.return_value = content
    return node


# ---------------------------------------------------------------------------
# _is_excluded()
# ---------------------------------------------------------------------------


class TestIsExcluded:
    def test_path_inside_node_modules_is_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/node_modules/lodash/index.js") is True

    def test_path_inside_venv_is_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/.venv/lib/python3.11/site.py") is True

    def test_path_inside_pycache_is_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/src/__pycache__/module.cpython-311.pyc") is True

    def test_path_inside_git_is_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/.git/config") is True

    def test_path_inside_rag_index_is_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/rag_index/docstore.json") is True

    def test_valid_py_file_is_not_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/src/api/auth.py") is False

    def test_valid_ts_file_is_not_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/frontend/src/App.tsx") is False

    def test_valid_sql_file_is_not_excluded(self) -> None:
        rag = _make_rag()
        assert rag._is_excluded("/fake/repo/migrations/001_init.sql") is False


# ---------------------------------------------------------------------------
# _detect_language()
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "file_path, expected",
        [
            ("/repo/main.py", "python"),
            ("/repo/app.ts", "typescript"),
            ("/repo/components/Button.tsx", "typescript"),
            ("/repo/index.js", "javascript"),
            ("/repo/App.jsx", "javascript"),
            ("/repo/schema.sql", "sql"),
            ("/repo/config.yaml", "yaml"),
            ("/repo/docker-compose.yml", "yaml"),
            ("/repo/pyproject.toml", "toml"),
            ("/repo/README.md", "markdown"),
        ],
    )
    def test_known_extension_returns_correct_language(
        self, file_path: str, expected: str
    ) -> None:
        rag = _make_rag()
        assert rag._detect_language(file_path) == expected

    def test_unknown_extension_returns_unknown(self) -> None:
        rag = _make_rag()
        assert rag._detect_language("/repo/binary.exe") == "unknown"

    def test_no_extension_returns_unknown(self) -> None:
        rag = _make_rag()
        assert rag._detect_language("/repo/Makefile") == "unknown"


# ---------------------------------------------------------------------------
# search() — mocked VectorStoreIndex
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_rag_with_index(self, nodes: list) -> CodebaseRAG:
        rag = _make_rag(min_score=0.3)
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        mock_index.as_retriever.return_value = mock_retriever
        mock_index.docstore.docs = {}
        rag._index = mock_index
        rag._indexed_files = ["/repo/a.py", "/repo/b.py"]
        return rag

    def test_returns_rag_search_result(self) -> None:
        nodes = [_make_node("/repo/auth.py", "def login(): pass", 0.9)]
        rag = self._make_rag_with_index(nodes)

        result = rag.search("OAuth login")

        assert isinstance(result, RAGSearchResult)
        assert result.query == "OAuth login"
        assert len(result.files) == 1

    def test_result_has_correct_structure(self) -> None:
        nodes = [_make_node("/repo/auth.py", "def login(): pass", 0.85)]
        rag = self._make_rag_with_index(nodes)

        result = rag.search("user authentication")
        f = result.files[0]

        assert f.file_name == "auth.py"
        assert f.file_path == "/repo/auth.py"
        assert f.score == 0.85
        assert f.language == "python"
        assert "login" in f.content

    def test_filters_results_below_min_score(self) -> None:
        nodes = [
            _make_node("/repo/auth.py", "auth code", 0.9),
            _make_node("/repo/utils.py", "util code", 0.2),  # below 0.3
            _make_node("/repo/models.py", "model code", 0.1),  # below 0.3
        ]
        rag = self._make_rag_with_index(nodes)

        result = rag.search("authentication")

        assert len(result.files) == 1
        assert result.files[0].file_path == "/repo/auth.py"

    def test_returns_total_indexed_count(self) -> None:
        nodes = [_make_node("/repo/auth.py", "code", 0.9)]
        rag = self._make_rag_with_index(nodes)
        rag._indexed_files = ["/repo/a.py", "/repo/b.py", "/repo/c.py"]

        result = rag.search("something")

        assert result.total_indexed == 3

    def test_search_duration_ms_is_positive(self) -> None:
        nodes = [_make_node("/repo/auth.py", "code", 0.9)]
        rag = self._make_rag_with_index(nodes)

        result = rag.search("anything")

        assert result.search_duration_ms >= 0

    def test_none_score_treated_as_zero(self) -> None:
        node = _make_node("/repo/file.py", "code", 0.0)
        node.score = None
        rag = self._make_rag_with_index([node])

        result = rag.search("query")

        # score=0.0 < min_score=0.3 → filtered out
        assert len(result.files) == 0


# ---------------------------------------------------------------------------
# search_by_type()
# ---------------------------------------------------------------------------


class TestSearchByType:
    def test_filters_to_python_files_only(self) -> None:
        rag = _make_rag(min_score=0.0)
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_node("/repo/auth.py", "python code", 0.9),
            _make_node("/repo/App.tsx", "react component", 0.85),
            _make_node("/repo/schema.sql", "sql schema", 0.8),
            _make_node("/repo/models.py", "models", 0.75),
        ]
        mock_index.as_retriever.return_value = mock_retriever
        mock_index.docstore.docs = {}
        rag._index = mock_index
        rag._indexed_files = []

        result = rag.search_by_type("auth", [".py"])

        assert all(f.language == "python" for f in result.files)
        assert len(result.files) == 2

    def test_filters_to_typescript_only(self) -> None:
        rag = _make_rag(min_score=0.0)
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_node("/repo/auth.py", "python", 0.9),
            _make_node("/repo/Button.tsx", "react", 0.8),
            _make_node("/repo/api.ts", "api client", 0.7),
        ]
        mock_index.as_retriever.return_value = mock_retriever
        mock_index.docstore.docs = {}
        rag._index = mock_index
        rag._indexed_files = []

        result = rag.search_by_type("component", [".ts", ".tsx"])

        assert all(f.language == "typescript" for f in result.files)


# ---------------------------------------------------------------------------
# get_file()
# ---------------------------------------------------------------------------


class TestGetFile:
    def _make_doc_node(self, file_path: str, content: str) -> MagicMock:
        node = MagicMock()
        node.metadata = {"file_path": file_path}
        node.get_content.return_value = content
        return node

    def test_returns_correct_relevant_file_for_known_path(self) -> None:
        rag = _make_rag()
        mock_index = MagicMock()
        mock_index.docstore.docs = {
            "node-1": self._make_doc_node("/repo/auth.py", "def login(): pass"),
        }
        rag._index = mock_index

        result = rag.get_file("/repo/auth.py")

        assert result is not None
        assert result.file_name == "auth.py"
        assert result.file_path == "/repo/auth.py"
        assert result.score == 1.0
        assert result.language == "python"
        assert "login" in result.content

    def test_returns_none_for_unknown_path(self) -> None:
        rag = _make_rag()
        mock_index = MagicMock()
        mock_index.docstore.docs = {
            "node-1": self._make_doc_node("/repo/auth.py", "code"),
        }
        rag._index = mock_index

        result = rag.get_file("/repo/nonexistent.py")

        assert result is None

    def test_returns_none_when_index_has_no_docs(self) -> None:
        rag = _make_rag()
        mock_index = MagicMock()
        mock_index.docstore.docs = {}
        rag._index = mock_index

        result = rag.get_file("/repo/anything.py")

        assert result is None


# ---------------------------------------------------------------------------
# index_stats()
# ---------------------------------------------------------------------------


class TestIndexStats:
    def test_returns_dict_with_required_keys(self) -> None:
        rag = _make_rag()
        rag._indexed_files = ["/repo/a.py", "/repo/b.py"]

        stats = rag.index_stats()

        assert "total_files" in stats
        assert "indexed_files" in stats
        assert "last_updated" in stats

    def test_total_files_matches_indexed_files_count(self) -> None:
        rag = _make_rag()
        rag._indexed_files = ["/repo/a.py", "/repo/b.py", "/repo/c.py"]

        stats = rag.index_stats()

        assert stats["total_files"] == 3
        assert stats["indexed_files"] == 3

    def test_last_updated_is_never_when_not_built(self) -> None:
        rag = _make_rag()

        stats = rag.index_stats()

        assert stats["last_updated"] == "never"

    def test_last_updated_is_iso_string_after_build(self) -> None:
        from datetime import datetime

        rag = _make_rag()
        rag._last_built = datetime(2026, 1, 15, 10, 0, 0)

        stats = rag.index_stats()

        assert "2026-01-15" in str(stats["last_updated"])


# ---------------------------------------------------------------------------
# create_rag() — env var config
# ---------------------------------------------------------------------------


class TestCreateRag:
    def test_reads_config_from_env_vars(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_REPO_LOCAL_PATH", "/my/repo")
        monkeypatch.setenv("RAG_INDEX_PATH", "/my/index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("RAG_TOP_K", "10")
        monkeypatch.setenv("RAG_MIN_SCORE", "0.5")

        rag = create_rag()

        assert rag._config.repo_path == "/my/repo"
        assert rag._config.index_path == "/my/index"
        assert rag._config.embedding_model == "text-embedding-ada-002"
        assert rag._config.top_k == 10
        assert rag._config.min_score == 0.5

    def test_explicit_repo_path_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_REPO_LOCAL_PATH", "/env/repo")

        rag = create_rag(repo_path="/explicit/repo")

        assert rag._config.repo_path == "/explicit/repo"

    def test_defaults_when_env_vars_absent(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_REPO_LOCAL_PATH", raising=False)
        monkeypatch.delenv("RAG_INDEX_PATH", raising=False)
        monkeypatch.delenv("RAG_TOP_K", raising=False)
        monkeypatch.delenv("RAG_MIN_SCORE", raising=False)

        rag = create_rag()

        assert rag._config.repo_path == "."
        assert rag._config.index_path == "./rag_index"
        assert rag._config.top_k == 6
        assert rag._config.min_score == 0.3


# ---------------------------------------------------------------------------
# _get_changed_files()
# ---------------------------------------------------------------------------


class TestGetChangedFiles:
    def test_returns_list_of_changed_file_paths(self, tmp_path) -> None:
        (tmp_path / "src").mkdir()
        py_file = tmp_path / "src" / "auth.py"
        py_file.write_text("code")

        rag = _make_rag(repo_path=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "src/auth.py\n"

        with patch("subprocess.run", return_value=mock_result):
            changed = rag._get_changed_files()

        assert any("auth.py" in p for p in changed)

    def test_excludes_non_indexed_extensions(self, tmp_path) -> None:
        (tmp_path / "src").mkdir()
        bin_file = tmp_path / "src" / "data.bin"
        bin_file.write_text("binary")

        rag = _make_rag(repo_path=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "src/data.bin\n"

        with patch("subprocess.run", return_value=mock_result):
            changed = rag._get_changed_files()

        assert changed == []

    def test_returns_empty_list_when_git_fails(self, tmp_path) -> None:
        rag = _make_rag(repo_path=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "not a git repository"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            changed = rag._get_changed_files()

        assert changed == []

    def test_returns_empty_list_on_subprocess_exception(self, tmp_path) -> None:
        rag = _make_rag(repo_path=str(tmp_path))

        with patch("subprocess.run", side_effect=Exception("git not found")):
            changed = rag._get_changed_files()

        assert changed == []

    def test_skips_excluded_dirs_in_changed_files(self, tmp_path) -> None:
        (tmp_path / "node_modules" / "lib").mkdir(parents=True)
        js_file = tmp_path / "node_modules" / "lib" / "index.js"
        js_file.write_text("module code")

        rag = _make_rag(repo_path=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "node_modules/lib/index.js\n"

        with patch("subprocess.run", return_value=mock_result):
            changed = rag._get_changed_files()

        assert changed == []
