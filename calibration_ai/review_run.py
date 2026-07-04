"""CLI for reviewing a saved calibration run with the NVIDIA Ising model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .nvidia_ising_client import NvidiaIsingClient
from .reviewer import CalibrationAIReviewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review saved calibration figures with NVIDIA Ising Calibration.")
    parser.add_argument("run_directory", type=Path, help="Saved calibration run directory.")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL. Defaults to NVIDIA hosted API.")
    parser.add_argument("--api-key", help="NVIDIA API key. Defaults to NVIDIA_API_KEY.")
    parser.add_argument("--local", action="store_true", help="Use http://localhost:8000/v1 for a local NIM.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context", action="append", default=[], help="Extra context as name=value.")
    return parser


def parse_context(items: list[str]) -> dict[str, str]:
    context = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --context name=value, got {item!r}")
        name, value = item.split("=", 1)
        context[name.strip()] = value.strip()
    return context


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = "http://localhost:8000/v1" if args.local else args.base_url
    client = NvidiaIsingClient(api_key=args.api_key, base_url=base_url)
    review = CalibrationAIReviewer(client).review_run(
        args.run_directory,
        extra_context=parse_context(args.context),
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                "ai_review": str(review.json_path),
                "markdown": str(review.markdown_path),
                "figures": [str(path) for path in review.figure_paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
