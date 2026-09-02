"""Write a review-only coarse DRAG-beta/kappa plan; never runs hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.drag_beta_kappa_calibration import coarse_plan, plan_sha256, write_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-qubit", required=True)
    parser.add_argument("--existing-kappa-mhz-inv", required=True, type=float)
    parser.add_argument("--include-no-correction", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = coarse_plan(
        target_qubit=args.target_qubit,
        existing_kappa_mhz_inv=args.existing_kappa_mhz_inv,
        include_no_correction=args.include_no_correction,
    )
    path = write_plan(plan, args.output)
    print(f"Review-only plan: {path}")
    print(f"Approval hash: {plan_sha256(plan)}")
    print("No hardware task was submitted.")


if __name__ == "__main__":
    main()
