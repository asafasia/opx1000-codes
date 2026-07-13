"""Small command-line interface for the local calibration registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .database import CalibrationResultsDatabase, DEFAULT_DATABASE_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the calibration-results SQLite registry.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the database schema.")
    history = commands.add_parser("history", help="Show a metric history for one target.")
    history.add_argument("target")
    history.add_argument("metric")
    history.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    database = CalibrationResultsDatabase(args.database)
    if args.command == "init":
        database.initialize()
        print(args.database)
    else:
        print(json.dumps(database.latest_metrics(args.target, args.metric, limit=args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
