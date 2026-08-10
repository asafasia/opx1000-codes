"""Restricted HTTP bridge for a VISA instrument attached by USB.

This file is intentionally standalone so it can be copied to the Windows
laptop that owns the USB connection.  It exposes only a small allowlist of
SCPI queries and never accepts VISA write commands.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

try:
    import pyvisa
except ImportError:  # pragma: no cover - exercised on the laptop at startup
    pyvisa = None


SAFE_QUERIES = frozenset(
    {
        "*IDN?",
        "*OPT?",
        "SYST:ERR?",
        "SYSTEM:ERROR?",
        "SYST:VERS?",
        "SYSTEM:VERSION?",
    }
)


def normalize_query(command: str) -> str:
    """Validate and normalize a query accepted by the restricted bridge."""
    normalized = command.strip().upper()
    if not normalized or len(normalized) > 128:
        raise ValueError("query must contain between 1 and 128 characters")
    if any(separator in normalized for separator in (";", "\r", "\n")):
        raise ValueError("compound or multiline SCPI commands are not allowed")
    if normalized not in SAFE_QUERIES:
        raise ValueError(
            "query is not on the read-only allowlist; allowed queries: "
            + ", ".join(sorted(SAFE_QUERIES))
        )
    return normalized


def normalize_usb_resource(resource: object) -> str:
    """Accept only USB VISA resource strings, never network or serial targets."""
    if not isinstance(resource, str):
        raise ValueError("resource must be a USB VISA resource string")
    normalized = resource.strip()
    if not normalized or len(normalized) > 512:
        raise ValueError("resource must be a non-empty USB VISA resource string")
    if not normalized.upper().startswith("USB"):
        raise ValueError("only USB VISA resources are allowed")
    return normalized


class VisaBridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        allowed_client: str,
        token: str,
        visa_backend: str | None = None,
    ) -> None:
        super().__init__(server_address, VisaBridgeHandler)
        self.allowed_client = allowed_client
        self.token = token
        self.visa_backend = visa_backend
        self.visa_lock = threading.Lock()

    def resource_manager(self):
        if pyvisa is None:
            raise RuntimeError("PyVISA is not installed; run: py -m pip install pyvisa")
        if self.visa_backend:
            return pyvisa.ResourceManager(self.visa_backend)
        return pyvisa.ResourceManager()


class VisaBridgeHandler(BaseHTTPRequestHandler):
    server: VisaBridgeServer

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), fmt % args)
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if self.client_address[0] != self.server.allowed_client:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "client IP is not allowed"})
            return False
        expected = f"Bearer {self.server.token}"
        if not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid bridge token"})
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > 4096:
            raise ValueError("request body must contain 1 to 4096 bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "mode": "read-only"})
            return
        if path == "/resources":
            try:
                with self.server.visa_lock:
                    manager = self.server.resource_manager()
                    try:
                        resources = [
                            resource
                            for resource in manager.list_resources()
                            if resource.upper().startswith("USB")
                        ]
                    finally:
                        manager.close()
                self._send_json(HTTPStatus.OK, {"resources": resources})
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": f"VISA resource discovery failed: {exc}"},
                )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not self._authorized():
            return
        if urlparse(self.path).path != "/query":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        try:
            payload = self._read_json()
            resource = normalize_usb_resource(payload.get("resource"))
            command = normalize_query(str(payload.get("command", "")))
            timeout_ms = int(payload.get("timeout_ms", 5000))
            if not 100 <= timeout_ms <= 30000:
                raise ValueError("timeout_ms must be between 100 and 30000")
        except (TypeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            with self.server.visa_lock:
                manager = self.server.resource_manager()
                try:
                    instrument = manager.open_resource(resource)
                    try:
                        instrument.timeout = timeout_ms
                        response = instrument.query(command)
                    finally:
                        instrument.close()
                finally:
                    manager.close()
            self._send_json(
                HTTPStatus.OK,
                {
                    "resource": resource,
                    "command": command,
                    "response": response.rstrip("\r\n"),
                },
            )
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"VISA query failed: {exc}"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="192.168.88.247", help="laptop LAN address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-client",
        default="192.168.88.250",
        help="the only client IP permitted to use the bridge",
    )
    parser.add_argument(
        "--token",
        help="authentication token; a random token is generated when omitted",
    )
    parser.add_argument(
        "--visa-backend",
        help="optional PyVISA backend such as @py; the installed IVI backend is used by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    token = args.token or secrets.token_urlsafe(24)
    server = VisaBridgeServer(
        (args.bind, args.port),
        allowed_client=args.allow_client,
        token=token,
        visa_backend=args.visa_backend,
    )
    print(f"VISA bridge listening on http://{args.bind}:{args.port}", flush=True)
    print(f"Allowed client: {args.allow_client}", flush=True)
    print(f"Token: {token}", flush=True)
    print("Mode: read-only allowlisted SCPI queries", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VISA bridge.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
