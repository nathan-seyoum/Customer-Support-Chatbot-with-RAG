"""CLI entry point: `support-rag-ingest <path>`.

Usage:
    python -m app.ingestion.cli data/docs
    support-rag-ingest data/docs   # after `pip install -e .`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the support knowledge base.")
    parser.add_argument("path", type=Path, help="File or directory of .md/.txt/.pdf documents")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s :: %(message)s")

    pipeline = IngestionPipeline()
    if args.reset:
        pipeline.store.reset()
        print("Collection reset.")

    report = pipeline.ingest_path(args.path)
    print(f"Ingested {report.documents} document(s) → {report.chunks} chunk(s).")
    print(f"Store now contains {pipeline.store.count()} chunk(s) total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
