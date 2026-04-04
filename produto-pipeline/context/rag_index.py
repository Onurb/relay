import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → language mapping
# ---------------------------------------------------------------------------

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RelevantFile(BaseModel):
    """A single file returned by a RAG search."""

    file_name: str
    file_path: str
    score: float
    content: str
    language: str


class RAGSearchResult(BaseModel):
    """The full result of a RAG search query."""

    query: str
    files: list[RelevantFile]
    total_indexed: int
    search_duration_ms: float


class RAGConfig(BaseModel):
    """Configuration for the CodebaseRAG index."""

    repo_path: str
    index_path: str = "./rag_index"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 6
    min_score: float = 0.3
    excluded_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            "dist",
            "build",
            ".next",
            "rag_index",
            "memory",
        ]
    )
    included_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".sql",
            ".yaml",
            ".yml",
            ".toml",
            ".md",
        ]
    )


# ---------------------------------------------------------------------------
# CodebaseRAG
# ---------------------------------------------------------------------------


class CodebaseRAG:
    """Shared context module that builds and queries a vector index of a repository.

    Used by the PromptEngineer and VibeCoder agents to automatically surface
    relevant files without manual path listing. Backed by LlamaIndex with
    OpenAI embeddings persisted to disk.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._index: VectorStoreIndex | None = None
        self._indexed_files: list[str] = []
        self._last_built: datetime | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Builds the vector index from scratch by walking the repository."""
        start = time.perf_counter()
        repo = Path(self._config.repo_path)

        valid_files = [
            str(p)
            for p in repo.rglob("*")
            if p.is_file()
            and not self._is_excluded(str(p))
            and p.suffix in self._config.included_extensions
        ]

        if not valid_files:
            logger.warning("No indexable files found in %s", repo)
            return

        reader = SimpleDirectoryReader(input_files=valid_files)
        documents = reader.load_data()

        embed_model = OpenAIEmbedding(model=self._config.embedding_model)
        self._index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model, show_progress=False
        )

        index_path = Path(self._config.index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(index_path))

        self._indexed_files = valid_files
        self._last_built = datetime.utcnow()
        duration = time.perf_counter() - start
        logger.info("RAG index built: %d files indexed in %.2fs", len(valid_files), duration)

    def update(self) -> None:
        """Rebuilds only the files changed since the last index build."""
        changed = self._get_changed_files()
        if not changed:
            logger.info("RAG index up to date")
            return

        self._ensure_index()
        assert self._index is not None

        embed_model = OpenAIEmbedding(model=self._config.embedding_model)
        reader = SimpleDirectoryReader(input_files=changed)
        documents = reader.load_data()

        for doc in documents:
            self._index.insert(doc)

        index_path = Path(self._config.index_path)
        self._index.storage_context.persist(persist_dir=str(index_path))

        self._last_built = datetime.utcnow()
        logger.info("RAG index updated: %d files refreshed", len(changed))

    def search(
        self, query: str, top_k: int | None = None
    ) -> RAGSearchResult:
        """Queries the index and returns the most relevant files."""
        self._ensure_index()
        assert self._index is not None

        k = top_k if top_k is not None else self._config.top_k
        start = time.perf_counter()

        retriever = self._index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)

        files: list[RelevantFile] = []
        for node in nodes:
            score = node.score if node.score is not None else 0.0
            if score < self._config.min_score:
                continue
            file_path = node.node.metadata.get("file_path", "")
            files.append(
                RelevantFile(
                    file_name=Path(file_path).name,
                    file_path=file_path,
                    score=round(score, 4),
                    content=node.node.get_content(),
                    language=self._detect_language(file_path),
                )
            )

        duration_ms = (time.perf_counter() - start) * 1000
        return RAGSearchResult(
            query=query,
            files=files,
            total_indexed=len(self._indexed_files),
            search_duration_ms=round(duration_ms, 2),
        )

    def search_by_type(
        self, query: str, extensions: list[str]
    ) -> RAGSearchResult:
        """Same as search() but restricts results to files with the given extensions."""
        result = self.search(query, top_k=self._config.top_k * 3)
        filtered = [
            f for f in result.files if Path(f.file_path).suffix in extensions
        ]
        return RAGSearchResult(
            query=result.query,
            files=filtered[: self._config.top_k],
            total_indexed=result.total_indexed,
            search_duration_ms=result.search_duration_ms,
        )

    def get_file(self, file_path: str) -> RelevantFile | None:
        """Returns a RelevantFile for a specific path, or None if not in the index."""
        self._ensure_index()
        assert self._index is not None

        target = str(Path(file_path))
        for node_id, node in self._index.docstore.docs.items():
            node_path = node.metadata.get("file_path", "")
            if str(Path(node_path)) == target:
                return RelevantFile(
                    file_name=Path(node_path).name,
                    file_path=node_path,
                    score=1.0,
                    content=node.get_content(),
                    language=self._detect_language(node_path),
                )
        return None

    def index_stats(self) -> dict[str, int | str]:
        """Returns basic statistics about the current index state."""
        return {
            "total_files": len(self._indexed_files),
            "indexed_files": len(self._indexed_files),
            "last_updated": (
                self._last_built.isoformat() if self._last_built else "never"
            ),
        }

    def setup_github_webhook(self) -> None:
        """Registers a GitHub webhook to trigger update() on each push to main.

        TODO: implement webhook server in a later sprint. For now this is a stub.
        The webhook endpoint will call self.update() when GitHub POSTs a push event.
        """
        logger.info(
            "setup_github_webhook() not yet implemented — "
            "RAG updates are triggered manually via update()."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_build(self) -> VectorStoreIndex:
        """Loads the index from disk if it exists, otherwise builds from scratch."""
        index_path = Path(self._config.index_path)
        if index_path.exists() and any(index_path.iterdir()):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_path)
                )
                index = load_index_from_storage(storage_context)
                logger.info("RAG index loaded from %s", index_path)
                return index  # type: ignore[return-value]
            except Exception as exc:
                logger.warning(
                    "Failed to load existing index (%s) — rebuilding.", exc
                )

        self.build()
        assert self._index is not None
        return self._index

    def _ensure_index(self) -> None:
        """Loads or builds the index if not already in memory."""
        if self._index is None:
            self._index = self._load_or_build()

    def _is_excluded(self, path: str) -> bool:
        """Returns True if any path component is in excluded_dirs."""
        parts = Path(path).parts
        return any(part in self._config.excluded_dirs for part in parts)

    def _detect_language(self, file_path: str) -> str:
        """Returns the language string for a file based on its extension."""
        ext = Path(file_path).suffix.lower()
        return _EXT_TO_LANGUAGE.get(ext, "unknown")

    def _get_changed_files(self) -> list[str]:
        """Returns files changed since the last commit using git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                cwd=self._config.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "git diff failed (%s) — falling back to full rebuild.",
                    result.stderr.strip(),
                )
                return []

            repo = Path(self._config.repo_path)
            changed: list[str] = []
            for rel_path in result.stdout.splitlines():
                rel_path = rel_path.strip()
                if not rel_path:
                    continue
                p = repo / rel_path
                if (
                    p.is_file()
                    and p.suffix in self._config.included_extensions
                    and not self._is_excluded(str(p))
                ):
                    changed.append(str(p))
            return changed

        except Exception as exc:
            logger.warning("Could not determine changed files: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_rag(repo_path: str | None = None) -> CodebaseRAG:
    """Creates a CodebaseRAG instance using environment variables as defaults.

    Args:
        repo_path: Path to the local repository root. Defaults to the
                   GITHUB_REPO_LOCAL_PATH env var, or "." if unset.
    """
    config = RAGConfig(
        repo_path=repo_path or os.getenv("GITHUB_REPO_LOCAL_PATH", "."),
        index_path=os.getenv("RAG_INDEX_PATH", "./rag_index"),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        top_k=int(os.getenv("RAG_TOP_K", "6")),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.3")),
    )
    return CodebaseRAG(config)
