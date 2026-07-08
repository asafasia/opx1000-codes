"""Run all selected scan domains for one cutoff region."""

from __future__ import annotations

from common import region_parser, run_region_from_args


def main() -> int:
    parser = region_parser(__doc__ or "")
    return run_region_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
