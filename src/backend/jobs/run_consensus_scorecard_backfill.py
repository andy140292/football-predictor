from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from backend.scorecard_service import run_consensus_scorecard_backfill
except ImportError:  # pragma: no cover
    from src.backend.scorecard_service import run_consensus_scorecard_backfill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consensus scorecard retrospective backfill.")
    parser.add_argument("--from-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--mode", default="national", help="Prediction mode (default: national)")
    parser.add_argument(
        "--model-version",
        default=os.getenv("MODEL_VERSION"),
        help="Model version label (default: MODEL_VERSION env)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_consensus_scorecard_backfill(
        from_date=args.from_date,
        to_date=args.to_date,
        mode=args.mode,
        model_version=args.model_version,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
