"""Local, read-only live waveform dashboard for the Tektronix oscilloscope.

The browser talks only to this local server. The server owns the VISA session,
which avoids browser CORS restrictions and keeps instrument access serialized.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import queue
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyvisa


DEFAULT_HOST = "192.168.88.251"
STATIC_DIR = Path(__file__).with_name("dashboard_static")
CHANNELS = (1, 2, 3, 4)
PREAMBLE_CACHE_SECONDS = 10.0


class RawSocketInstrument:
    """Small SCPI adapter for the TekVISA Socket Server."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self._socket = socket.create_connection((host, port), timeout=timeout_ms / 1_000)
        self._socket.settimeout(timeout_ms / 1_000)
        self.encoding = "latin-1"
        self.chunk_size = 1_048_576

    @property
    def timeout(self) -> int:
        return int((self._socket.gettimeout() or 0) * 1_000)

    @timeout.setter
    def timeout(self, timeout_ms: int) -> None:
        self._socket.settimeout(timeout_ms / 1_000)

    def close(self) -> None:
        self._socket.close()

    def write(self, command: str) -> None:
        self._socket.sendall(command.encode(self.encoding) + b"\n")

    def query(self, command: str) -> str:
        self.write(command)
        return self._read_until_newline().decode(self.encoding)

    def read_raw(self) -> bytes:
        prefix = self._read_exact(2)
        if prefix[:1] != b"#" or not prefix[1:2].isdigit():
            raise ValueError("scope socket returned an invalid waveform block")
        digits = int(prefix[1:2])
        size_text = self._read_exact(digits)
        payload = self._read_exact(int(size_text))

        # The socket server normally appends a line terminator after the IEEE
        # block. Consume it when immediately available without delaying frames.
        old_timeout = self._socket.gettimeout()
        try:
            self._socket.settimeout(0.01)
            try:
                self._socket.recv(2)
            except TimeoutError:
                pass
        finally:
            self._socket.settimeout(old_timeout)
        return prefix + size_text + payload

    def _read_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._socket.recv(min(self.chunk_size, size - len(chunks)))
            if not chunk:
                raise ConnectionError("scope socket closed during waveform transfer")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_until_newline(self) -> bytes:
        chunks = bytearray()
        while True:
            byte = self._socket.recv(1)
            if not byte:
                raise ConnectionError("scope socket closed while reading a response")
            chunks.extend(byte)
            if byte == b"\n":
                return bytes(chunks)


def parse_ieee_block(raw: bytes) -> np.ndarray:
    """Return the uint8 payload from an IEEE 488.2 definite-length block."""
    if len(raw) < 3 or raw[:1] != b"#" or not raw[1:2].isdigit():
        raise ValueError("scope returned an invalid binary waveform block")
    digits = int(raw[1:2])
    if digits == 0 or len(raw) < 2 + digits:
        raise ValueError("scope returned an unsupported binary waveform block")
    payload_size = int(raw[2 : 2 + digits])
    start = 2 + digits
    end = start + payload_size
    if len(raw) < end:
        raise ValueError(f"incomplete waveform: expected {payload_size} bytes")
    return np.frombuffer(raw[start:end], dtype=np.uint8).copy()


@dataclass
class ScopeConnection:
    host: str = DEFAULT_HOST
    timeout_ms: int = 5_000
    transport: str = "socket"
    socket_port: int = 4_000

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._resource_manager = None
        self._scope = None
        self._identity = None
        self._last_error = None
        self._transfer_key = None
        self._enabled_channels: set[int] = set()
        self._preamble_cache: dict[int, tuple[float, dict[str, float]]] = {}

    @property
    def resource_name(self) -> str:
        if self.transport == "socket":
            return f"TCPIP::{self.host}::{self.socket_port}::SOCKET"
        return f"TCPIP::{self.host}::INSTR"

    def close(self) -> None:
        with self._lock:
            if self._scope is not None:
                try:
                    self._scope.close()
                finally:
                    self._scope = None
            if self._resource_manager is not None:
                self._resource_manager.close()
                self._resource_manager = None

    def _disconnect(self, message: str) -> None:
        self._last_error = message
        if self._scope is not None:
            try:
                self._scope.close()
            except Exception:
                pass
        self._scope = None
        self._identity = None
        self._transfer_key = None
        self._enabled_channels.clear()
        self._preamble_cache.clear()
        if self._resource_manager is not None:
            try:
                self._resource_manager.close()
            except Exception:
                pass
        self._resource_manager = None

    def _connect(self):
        if self._scope is not None:
            return self._scope
        if self.transport == "socket":
            scope = RawSocketInstrument(self.host, self.socket_port, self.timeout_ms)
            scope.encoding = "latin-1"
            scope.timeout = self.timeout_ms
            scope.chunk_size = 1_048_576
            self._identity = scope.query("*IDN?").strip()
            self._scope = scope
            self._last_error = None
            return scope
        if self._resource_manager is None:
            self._resource_manager = pyvisa.ResourceManager()
        scope = self._resource_manager.open_resource(self.resource_name)
        scope.encoding = "latin-1"
        scope.timeout = self.timeout_ms
        scope.chunk_size = 1_048_576
        self._identity = scope.query("*IDN?").strip()
        self._scope = scope
        self._last_error = None
        return scope

    def status(self) -> dict:
        with self._lock:
            try:
                self._connect()
                return {
                    "connected": True,
                    "host": self.host,
                    "identity": self._identity,
                    "resource": self.resource_name,
                }
            except Exception as exc:
                self._disconnect(str(exc))
                return {
                    "connected": False,
                    "host": self.host,
                    "identity": None,
                    "resource": self.resource_name,
                    "error": self._last_error,
                }

    def acquire(self, channels: list[int], max_points: int) -> dict:
        started = time.perf_counter()
        with self._lock:
            try:
                scope = self._connect()
                traces = [self._acquire_channel(scope, ch, max_points) for ch in channels]
            except Exception as exc:
                self._disconnect(str(exc))
                raise RuntimeError(str(exc)) from exc

        return {
            "identity": self._identity,
            "host": self.host,
            "captured_at": time.time(),
            "duration_ms": round((time.perf_counter() - started) * 1_000, 1),
            "traces": traces,
        }

    def _acquire_channel(self, scope, channel: int, max_points: int) -> dict:
        # A Tektronix channel must be displayed before CURVE? will return its
        # waveform. Selecting it in the dashboard therefore enables only its
        # display; no scale, coupling, trigger, or acquisition settings change.
        if channel not in self._enabled_channels:
            scope.write(f"SELECT:CH{channel} ON")
            self._enabled_channels.add(channel)

        transfer_key = (channel, max_points)
        if transfer_key != self._transfer_key:
            scope.write(f"DATA:SOURCE CH{channel}")
            scope.write("DATA:WIDTH 1")
            scope.write("DATA:ENC RPB")
            scope.write("DATA:START 1")
            scope.write(f"DATA:STOP {max_points}")
            self._transfer_key = transfer_key

        cached = self._preamble_cache.get(channel)
        if cached is None or time.monotonic() - cached[0] >= PREAMBLE_CACHE_SECONDS:
            preamble = {
                "x_increment": float(scope.query("WFMOUTPRE:XINCR?")),
                "x_zero": float(scope.query("WFMOUTPRE:XZERO?")),
                "y_multiplier": float(scope.query("WFMOUTPRE:YMULT?")),
                "y_zero": float(scope.query("WFMOUTPRE:YZERO?")),
                "y_offset": float(scope.query("WFMOUTPRE:YOFF?")),
            }
            self._preamble_cache[channel] = (time.monotonic(), preamble)
        else:
            preamble = cached[1]

        x_increment = preamble["x_increment"]
        x_zero = preamble["x_zero"]
        y_multiplier = preamble["y_multiplier"]
        y_zero = preamble["y_zero"]
        y_offset = preamble["y_offset"]

        scope.write("CURVE?")
        adc = parse_ieee_block(scope.read_raw())
        voltage = (adc.astype(np.float64) - y_offset) * y_multiplier + y_zero
        finite = voltage[np.isfinite(voltage)]
        if finite.size == 0:
            raise ValueError(f"CH{channel} returned no finite samples")

        # Keep the instrument request identical to the proven standalone
        # script. Reduce only the JSON/plot payload after the full waveform is
        # received, preserving the time axis with the adjusted increment.
        stride = max(1, math.ceil(voltage.size / max_points))
        displayed_voltage = voltage[::stride]

        return {
            "channel": channel,
            "x_zero": x_zero,
            "x_increment": x_increment * stride,
            "values": displayed_voltage.tolist(),
            "metrics": {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "pk_pk": float(np.ptp(finite)),
                "rms": float(math.sqrt(np.mean(np.square(finite)))),
                "points": int(voltage.size),
                "sample_rate": 1.0 / x_increment if x_increment > 0 else None,
            },
        }


class DashboardHandler(SimpleHTTPRequestHandler):
    server: "DashboardServer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json(HTTPStatus.OK, self.server.status_payload())
            return
        if parsed.path == "/api/waveforms":
            try:
                query = parse_qs(parsed.query)
                raw_channels = query.get("channels", ["1"])[0]
                channels = sorted({int(value) for value in raw_channels.split(",")})
                if not channels or any(channel not in CHANNELS for channel in channels):
                    raise ValueError("channels must be a comma-separated subset of 1,2,3,4")
                max_points = int(query.get("max_points", ["5000"])[0])
                if not 100 <= max_points <= 50_000:
                    raise ValueError("max_points must be between 100 and 50000")
                self.server.request_capture(channels, max_points)
                capture = self.server.current_capture(channels)
                if capture is None:
                    error = self.server.current_error()
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": error or "Waiting for the first instrument capture"},
                    )
                else:
                    self._json(HTTPStatus.OK, capture)
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": str(exc), "host": self.server.scope.host},
                )
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address, scope: ScopeConnection):
        super().__init__(address, DashboardHandler)
        self.scope = scope
        self._state_lock = threading.Lock()
        self._channels = [1]
        self._max_points = 2_000
        self._capture = None
        self._error = None

    def request_capture(self, channels: list[int], max_points: int) -> None:
        with self._state_lock:
            self._channels = list(channels)
            self._max_points = max_points

    def requested_capture(self) -> tuple[list[int], int]:
        with self._state_lock:
            return list(self._channels), self._max_points

    def publish_capture(self, capture: dict) -> None:
        with self._state_lock:
            self._capture = capture
            self._error = None

    def publish_error(self, message: str) -> None:
        with self._state_lock:
            self._error = message

    def current_capture(self, channels: list[int]) -> dict | None:
        with self._state_lock:
            if self._capture is None:
                return None
            captured_channels = [trace["channel"] for trace in self._capture["traces"]]
            return self._capture if captured_channels == channels else None

    def current_error(self) -> str | None:
        with self._state_lock:
            return self._error

    def status_payload(self) -> dict:
        with self._state_lock:
            return {
                "connected": self._capture is not None and self._error is None,
                "host": self.scope.host,
                "identity": self._capture.get("identity") if self._capture else None,
                "resource": self.scope.resource_name,
                "error": self._error,
            }


def acquisition_worker(
    config_queue,
    result_queue,
    stop_event,
    host: str,
    timeout_ms: int,
    transport: str,
    socket_port: int,
) -> None:
    """Run NI-VISA in an isolated, single-threaded process.

    This mirrors ``scope/live.py``. Older TekVISA versions used by the
    MSO70804C can hang when VISA calls share a process with HTTP threads.
    """
    scope = ScopeConnection(host, timeout_ms, transport, socket_port)
    channels = [1]
    max_points = 2_000
    try:
        while not stop_event.is_set():
            try:
                while True:
                    channels, max_points = config_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                result_queue.put(("capture", scope.acquire(channels, max_points)))
                # A short yield prevents a tight loop while keeping steady-state
                # capture responsive once the preamble is cached.
                time.sleep(0.05)
            except RuntimeError as exc:
                result_queue.put(("error", str(exc)))
                time.sleep(1.0)
    finally:
        scope.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-host", default=DEFAULT_HOST)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-ms", type=int, default=5_000)
    parser.add_argument(
        "--transport",
        choices=("socket", "vxi11"),
        default="socket",
        help="Instrument transport (default: raw TekVISA socket; no automatic fallback)",
    )
    parser.add_argument("--socket-port", type=int, default=4_000)
    args = parser.parse_args()

    scope = ScopeConnection(args.scope_host, args.timeout_ms, args.transport, args.socket_port)
    server = DashboardServer((args.bind, args.port), scope)
    process_context = multiprocessing.get_context("spawn")
    config_queue = process_context.Queue(maxsize=4)
    result_queue = process_context.Queue(maxsize=2)
    stop_event = process_context.Event()
    worker = process_context.Process(
        target=acquisition_worker,
        args=(
            config_queue,
            result_queue,
            stop_event,
            args.scope_host,
            args.timeout_ms,
            args.transport,
            args.socket_port,
        ),
        daemon=True,
    )
    worker.start()
    print(f"Oscilloscope dashboard: http://{args.bind}:{args.port}", flush=True)
    print(f"Instrument: {scope.resource_name}", flush=True)
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    last_requested = None
    try:
        while True:
            requested = server.requested_capture()
            if requested != last_requested:
                try:
                    config_queue.put_nowait(requested)
                    last_requested = requested
                except queue.Full:
                    pass
            try:
                kind, payload = result_queue.get(timeout=0.25)
                if kind == "capture":
                    server.publish_capture(payload)
                else:
                    server.publish_error(payload)
            except queue.Empty:
                if not worker.is_alive():
                    server.publish_error("Oscilloscope acquisition process stopped unexpectedly")
                    break
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        server.shutdown()
        http_thread.join(timeout=2)
        stop_event.set()
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
