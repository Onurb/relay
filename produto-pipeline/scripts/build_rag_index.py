"""Standalone script to build, update, inspect, or query the codebase RAG index.

Usage:
    python scripts/build_rag_index.py --build
    python scripts/build_rag_index.py --update
    python scripts/build_rag_index.py --stats
    python scripts/build_rag_index.py --search "OAuth2 with Google"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running the script directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from context.rag_index import create_rag

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_build(rag) -> None:
    """Performs a full index rebuild from scratch."""
    logger.info("Building RAG index from scratch…")
    rag.build()
    stats = rag.index_stats()
    print(f"Done. {stats['indexed_files']} files indexed.")


def cmd_update(rag) -> None:
    """Incrementally updates the index with changed files only."""
    logger.info("Updating RAG index (incremental)…")
    rag.update()
    stats = rag.index_stats()
    print(f"Done. Index contains {stats['indexed_files']} files.")


def cmd_stats(rag) -> None:
    """Prints current index statistics."""
    stats = rag.index_stats()
    print("RAG Index Statistics")
    print("--------------------")
    print(f"Total files indexed : {stats['total_files']}")
    print(f"Last updated        : {stats['last_updated']}")
    print(f"Index path          : {rag._config.index_path}")
    print(f"Repo path           : {rag._config.repo_path}")
    print(f"Embedding model     : {rag._config.embedding_model}")
    print(f"top_k               : {rag._config.top_k}")
    print(f"min_score           : {rag._config.min_score}")


def cmd_search(rag, query: str) -> None:
    """Runs a similarity search and prints the top results."""
    logger.info("Searching for: %s", query)
    result = rag.search(query)

    print(f'\nQuery: "{result.query}"')
    if not result.files:
        print("No results found above the minimum score threshold.")
        return

    print(
        f"Results ({len(result.files)} files, "
        f"{result.search_duration_ms:.0f}ms):"
    )
    for i, f in enumerate(result.files, start=1):
        print(f"  {i}. {f.file_path} (score: {f.score:.2f}) [{f.language}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, update, or query the codebase RAG index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build",
        action="store_true",
        help="Perform a full index rebuild from scratch.",
    )
    group.add_argument(
        "--update",
        action="store_true",
        help="Incrementally update the index with changed files only.",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Print current index statistics.",
    )
    group.add_argument(
        "--search",
        metavar="QUERY",
        help="Run a similarity search against the index.",
    )
    parser.add_argument(
        "--repo",
        metavar="PATH",
        default=None,
        help="Path to the repository root (overrides GITHUB_REPO_LOCAL_PATH).",
    )

    args = parser.parse_args()
    rag = create_rag(repo_path=args.repo)

    if args.build:
        cmd_build(rag)
    elif args.update:
        cmd_update(rag)
    elif args.stats:
        cmd_stats(rag)
    elif args.search:
        cmd_search(rag, args.search)


if __name__ == "__main__":
    main()
