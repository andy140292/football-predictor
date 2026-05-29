from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from backend.matches_admin_service import cleanup_matches_duplicates
except ImportError:  # pragma: no cover
    from src.backend.matches_admin_service import cleanup_matches_duplicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean duplicate rows from the matches table.")
    parser.add_argument("--from-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to-date", required=True, help="End date in YYYY-MM-DD format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = cleanup_matches_duplicates(from_date=args.from_date, to_date=args.to_date)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
