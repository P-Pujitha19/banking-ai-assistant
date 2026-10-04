#!/usr/bin/env python
"""Build the RAG vector index: ``python scripts/build_vector_store.py [--rebuild]``."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking_ai.config.settings import get_settings  # noqa: E402
from banking_ai.rag.service import RagService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the banking knowledge-base vector store")
    parser.add_argument("--rebuild", action="store_true", help="force a full rebuild")
    parser.add_argument("--query", help="optional test query to run after building")
    args = parser.parse_args()
    rag = RagService(get_settings()).ensure_index(rebuild=args.rebuild)
    print(f"Vector store ready: backend={rag.backend}, chunks={len(rag.chunks)}, location={rag.settings.vector_store_dir}")
    if args.query:
        for hit in rag.retrieve(args.query).chunks:
            print(f"  {hit.score:.3f}  {hit.title} / {hit.section}")


if __name__ == "__main__":
    main()
