"""Ingest manual text or CSV discussions from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.database.session import create_database_engine, create_session_factory
from src.ingestion.manual import manual_submission, parse_csv_submissions
from src.services.discovery_service import build_discovery_service


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""

    parser = argparse.ArgumentParser(description="Ingest discussions into InSift.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One manually submitted discussion.")
    source.add_argument("--csv", type=Path, help="Path to a UTF-8 source CSV.")
    parser.add_argument("--source-url")
    parser.add_argument("--title")
    parser.add_argument("--author")
    parser.add_argument("--community")
    parser.add_argument("--max-rows", type=int, default=100)
    return parser


def main() -> None:
    """Run ingestion and print a compact outcome summary."""

    args = build_parser().parse_args()
    if args.text:
        submissions = [
            manual_submission(
                args.text,
                source_url=args.source_url,
                title=args.title,
                source_author=args.author,
                community=args.community,
            )
        ]
    else:
        submissions = parse_csv_submissions(args.csv.read_bytes(), max_rows=args.max_rows)

    settings = get_settings()
    SessionFactory = create_session_factory(create_database_engine(settings))
    with SessionFactory() as session:
        results = build_discovery_service(session, settings).process_many(submissions)
    accepted = sum(result.accepted and not result.duplicate for result in results)
    rejected = sum(not result.accepted and not result.duplicate for result in results)
    duplicates = sum(result.duplicate for result in results)
    print(
        f"Processed {len(results)} discussion(s): {accepted} accepted, "
        f"{rejected} rejected, {duplicates} duplicate(s)."
    )


if __name__ == "__main__":
    main()
