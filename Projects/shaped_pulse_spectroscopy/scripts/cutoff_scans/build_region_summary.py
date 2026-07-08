"""Build combined CSV summaries from an existing cutoff-region campaign folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import build_region_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_dir", type=Path)
    args = parser.parse_args()
    build_region_summary(args.region_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
