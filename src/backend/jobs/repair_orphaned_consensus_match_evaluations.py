from __future__ import annotations

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
    from backend.scorecard_service import repair_orphaned_consensus_match_evaluations
except ImportError:  # pragma: no cover
    from src.backend.scorecard_service import repair_orphaned_consensus_match_evaluations


def main() -> int:
    summary = repair_orphaned_consensus_match_evaluations()
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
