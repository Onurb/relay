# CLAUDE.md — Conventions for this repository

This file is read automatically by Claude Code at the start of every session.
Follow all conventions below without exception.

## Language

- All code, comments, docstrings, variable names, log messages, and commit messages
  must be written in **English only**.

## Code style

- Formatter and linter: **ruff** (`ruff check` + `ruff format`)
- Type checker: **mypy** (strict where possible)
- All functions and methods must have **type hints** on all parameters and return types.
- Never use `Any` unless absolutely unavoidable — document the reason inline.
- Maximum line length: 100 characters.
- Use f-strings for string interpolation; avoid `%`-style or `.format()`.

## Logging

- Use `logging.getLogger(__name__)` — never `print()` in library code.
- Log at `INFO` for normal operations, `WARNING` for recoverable issues,
  `ERROR` for failures that affect correctness.
- Never log secrets, tokens, or API keys.

## Pydantic models

- All inter-agent data (inputs, outputs, configs) must be typed as **Pydantic `BaseModel`**.
- Use `Field(default_factory=...)` for mutable defaults.
- Use `model_dump(mode="json")` when serialising to dicts for external calls.

## No hardcoded secrets

- All tokens, API keys, and credentials must come from **environment variables**.
- Use `python-dotenv` to load `.env` files in entrypoints.
- Add every new env var to `.env.example` with an empty value.

## Testing

- All new code must have unit tests in `tests/`.
- Use **pytest** and **unittest.mock**.
- Never make real API calls, file system writes, or subprocess calls in tests.
- Mock at the boundary: LLM clients, HTTP clients, subprocess, file I/O.

## How to run locally

```bash
cd produto-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in required values
python main.py
```

## RAG index

- The RAG index lives in `./rag_index/` — **never commit this directory** (already in `.gitignore`).
- Rebuild the index after cloning: `python scripts/build_rag_index.py --build`
- The index updates automatically on each pipeline run via `CodebaseRAG.update()`.
- To test a query manually: `python scripts/build_rag_index.py --search "your query"`
- `min_score` is `0.3` by default — increase it (`RAG_MIN_SCORE=0.6`) if results are too noisy.
- The index is stored in the path set by `RAG_INDEX_PATH` (default: `./rag_index`).
