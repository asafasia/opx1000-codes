"""Run one cutoff region/domain combination."""

from __future__ import annotations

from common import domain_parser, run_domain_from_args


def main() -> int:
    parser = domain_parser(__doc__ or "")
    return run_domain_from_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
