"""Command-line client for the restricted USB VISA bridge."""

from __future__ import annotations

import argparse
import getpass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    token: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=35) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Bridge returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"Could not reach VISA bridge: {exc.reason}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://192.168.88.247:8765")
    parser.add_argument("--token", help="token printed by the laptop bridge")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("health")
    subparsers.add_parser("resources")
    query = subparsers.add_parser("query")
    query.add_argument("resource")
    query.add_argument("command", choices=["*IDN?", "*OPT?", "SYST:ERR?", "SYST:VERS?"])
    query.add_argument("--timeout-ms", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or getpass.getpass("Bridge token: ")
    if args.operation == "health":
        result = request_json(args.server, token, "/health")
    elif args.operation == "resources":
        result = request_json(args.server, token, "/resources")
    else:
        result = request_json(
            args.server,
            token,
            "/query",
            {
                "resource": args.resource,
                "command": args.command,
                "timeout_ms": args.timeout_ms,
            },
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
