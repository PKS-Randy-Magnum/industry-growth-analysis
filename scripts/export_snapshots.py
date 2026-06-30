#!/usr/bin/env python3
"""CLI: export snapshots from existing SQLite database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.snapshot_export import export_all
from src.db.load_sqlite import DB_PATH
from src.etl.parse_crosswalk import load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Export snapshot CSVs from SQLite")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--start", default="2019-Q1")
    parser.add_argument("--end", default="2026-Q1")
    args = parser.parse_args()
    manifest = export_all(args.db, load_registry(), args.start, args.end)
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
