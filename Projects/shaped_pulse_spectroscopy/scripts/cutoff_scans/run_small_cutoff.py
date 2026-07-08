"""Run or plan the small-cutoff campaign."""

from __future__ import annotations

import sys

from common import region_parser, run_region_from_args


def main() -> int:
    parser = region_parser(__doc__ or "")
    args = parser.parse_args(["--region", "small_cutoff", *sys.argv[1:]])
    return run_region_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
