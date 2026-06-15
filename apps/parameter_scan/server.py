"""Small local app for running and watching long parameter scans."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import mimetypes
import os
import sys
import threading
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parameter_scans.runner import ExperimentSpec, LongScanRunner, ScanConfig

CALIBRATIONS_ROOT = PROJECT_ROOT / "calibrations"
SINGLE_QUBIT_PROFILE_ROOT = PROJECT_ROOT / "profiles" / "single_qubit"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "parameter_scans"
PREFERRED_EXPERIMENTS = {
    "03a_qubit_spectroscopy.py",
    "05_T1.py",
    "06a_ramsey.py",
    "07_iq_blobs.py",
}
LIVE_SCAN_EXPERIMENTS = {
    "03a_qubit_spectroscopy.py": "Qubit spectroscopy",
    "05_T1.py": "T1",
    "06a_ramsey.py": "T2 Ramsey",
    "07_iq_blobs.py": "IQ blobs",
}
TERMINAL_OUTPUT_LIMIT = 120_000


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def experiment_scripts() -> list[dict[str, Any]]:
    scripts = []
    for path in sorted(CALIBRATIONS_ROOT.glob("*.py"), key=lambda item: item.name.lower()):
        if path.name not in LIVE_SCAN_EXPERIMENTS:
            continue
        scripts.append(
            {
                "name": LIVE_SCAN_EXPERIMENTS[path.name],
                "script": relative(path),
                "preferred": path.name in PREFERRED_EXPERIMENTS,
            }
        )
    return scripts


def available_qubits() -> list[str]:
    qubits_path = SINGLE_QUBIT_PROFILE_ROOT / "qubits.json"
    payload = read_json(qubits_path) or {}
    qubits = payload.get("qubits", {})
    if not isinstance(qubits, dict):
        return []
    return sorted(qubits, key=lambda name: int(name[1:]) if name.startswith("q") and name[1:].isdigit() else name)


def parse_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def read_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    summary_path = path / "summary.csv"
    if not summary_path.is_file():
        return []
    rows = []
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                value = parse_float(row.get("value"))
                timestamp = parse_time(row.get("timestamp", ""))
                rows.append(
                    {
                        **row,
                        "value": value,
                        "timestamp_epoch": timestamp.timestamp() if timestamp else None,
                    }
                )
    except OSError:
        return rows
    return rows


def group_series(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("value") is None:
            continue
        key = (str(row.get("experiment_name", "")), str(row.get("qubit", "")), str(row.get("parameter", "")))
        series = groups.setdefault(
            key,
            {
                "experiment_name": key[0],
                "qubit": key[1],
                "parameter": key[2],
                "unit": row.get("unit", ""),
                "points": [],
            },
        )
        series["points"].append(
            {
                "timestamp": row.get("timestamp", ""),
                "timestamp_epoch": row.get("timestamp_epoch"),
                "cycle": int(row.get("cycle") or 0),
                "value": row.get("value"),
                "success": row.get("success"),
            }
        )
    for series in groups.values():
        series["points"].sort(key=lambda point: (point["timestamp_epoch"] or 0, point["cycle"]))
        series["analysis"] = analyze_points(series["points"])
    return sorted(groups.values(), key=lambda item: (item["experiment_name"], item["qubit"], item["parameter"]))


def analyze_points(points: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([point["value"] for point in points if point.get("value") is not None], dtype=float)
    times = np.asarray(
        [point["timestamp_epoch"] for point in points if point.get("timestamp_epoch") is not None],
        dtype=float,
    )
    if values.size == 0:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    latest = float(values[-1])
    previous = float(values[-2]) if values.size > 1 else latest
    elapsed_hours = 0.0
    drift_per_hour = 0.0
    if values.size > 1 and times.size == values.size and times[-1] > times[0]:
        hours = (times - times[0]) / 3600.0
        elapsed_hours = float(hours[-1])
        drift_per_hour = float(np.polyfit(hours, values, 1)[0]) if elapsed_hours > 0 else 0.0
    relative_std = abs(std / mean) if mean else 0.0
    last_step_sigma = abs((latest - previous) / std) if std > 0 and values.size > 2 else 0.0
    if values.size < 3:
        verdict = "collecting"
    elif last_step_sigma >= 3:
        verdict = "jump"
    elif relative_std < 0.01:
        verdict = "stable"
    elif relative_std < 0.05:
        verdict = "watch"
    else:
        verdict = "drifting"
    return {
        "count": int(values.size),
        "latest": latest,
        "mean": mean,
        "std": std,
        "variance": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "relative_std": relative_std,
        "last_change": float(latest - previous),
        "last_step_sigma": last_step_sigma,
        "elapsed_hours": elapsed_hours,
        "drift_per_hour": drift_per_hour,
        "verdict": verdict,
    }


class ScanJob:
    """Thread-safe wrapper around one active scan run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.runner: LongScanRunner | None = None
        self.thread: threading.Thread | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self.terminal_output = ""

    def append_terminal(self, text: str) -> None:
        if not text:
            return
        with self.lock:
            self.terminal_output = (self.terminal_output + text)[-TERMINAL_OUTPUT_LIMIT:]

    def start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("A parameter scan is already running.")
            scripts = [str(item) for item in payload.get("experiments", []) if str(item).strip()]
            if not scripts:
                raise ValueError("Choose at least one experiment.")
            repetitions = int(payload.get("repetitions") or 1)
            interval_seconds = float(payload.get("interval_seconds") or 0)
            config = ScanConfig(
                name=str(payload.get("name") or "live_scan"),
                experiments=[ExperimentSpec(script=Path(script)) for script in scripts],
                repetitions=None if repetitions <= 0 else repetitions,
                interval_seconds=max(0.0, interval_seconds),
                continue_on_error=bool(payload.get("continue_on_error", False)),
                save_full_results=bool(payload.get("save_full_results", False)),
                profile_name="single_qubit",
                qubit=str(payload.get("qubit") or "q1"),
                output_root=DEFAULT_OUTPUT_ROOT,
            )
            self.runner = LongScanRunner(config, repository_root=PROJECT_ROOT)
            self.started_at = datetime.now().astimezone().isoformat()
            self.finished_at = None
            self.error = None
            self.terminal_output = (
                f"Starting parameter scan at {self.started_at}\n"
                f"Profile: single_qubit | Qubit: {config.qubit}\n"
                "Experiments:\n"
                + "".join(f"  - {spec.script.as_posix()}\n" for spec in config.experiments)
                + "\n"
            )
            self.thread = threading.Thread(target=self._run, name="parameter-scan-runner", daemon=True)
            self.thread.start()
            return self.status()

    def _run(self) -> None:
        try:
            assert self.runner is not None
            capture = TerminalCapture(self)
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                self.runner.run()
        except Exception as exc:  # Keep the control server alive.
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            self.append_terminal(f"\n{self.error}\n")
        finally:
            with self.lock:
                self.finished_at = datetime.now().astimezone().isoformat()
            self.append_terminal(f"\nFinished parameter scan at {self.finished_at}\n")

    def stop(self) -> dict[str, Any]:
        with self.lock:
            if not self.runner:
                return self.status()
            (self.runner.run_directory / self.runner.config.stop_file).write_text(
                datetime.now().astimezone().isoformat() + "\n",
                encoding="utf-8",
            )
            return self.status()

    def status(self) -> dict[str, Any]:
        runner = self.runner
        thread = self.thread
        run_directory = runner.run_directory if runner else None
        status_payload = read_json(run_directory / "status.json") if run_directory else None
        rows = read_summary(run_directory)
        errors = [row for row in rows if row.get("status") == "error"]
        return {
            "running": bool(thread and thread.is_alive()),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "run_directory": str(run_directory) if run_directory else None,
            "run_path": relative(run_directory) if run_directory else None,
            "status": status_payload or {},
            "records": rows,
            "series": group_series(rows),
            "errors": errors[-5:],
            "terminal_output": self.terminal_output,
        }


JOB = ScanJob()


class TerminalCapture:
    """File-like stdout/stderr sink for live terminal output."""

    def __init__(self, job: ScanJob) -> None:
        self.job = job

    def write(self, text: str) -> int:
        self.job.append_terminal(text)
        return len(text)

    def flush(self) -> None:
        return None


class ParameterScanHandler(SimpleHTTPRequestHandler):
    """Serve the live scan app and its JSON API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT / "static"), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/experiments":
                self.send_json({"experiments": experiment_scripts(), "qubits": available_qubits()})
                return
            if parsed.path == "/api/status":
                self.send_json(JOB.status())
                return
            if parsed.path == "/api/file":
                query = urllib.parse.parse_qs(parsed.query)
                self.serve_project_file(query.get("path", [""])[0])
                return
            super().do_GET()
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/start":
                self.send_json(JOB.start(self.read_json_body()))
                return
            if self.path == "/api/stop":
                self.send_json(JOB.stop())
                return
            self.send_json({"error": "Unknown endpoint."}, HTTPStatus.NOT_FOUND)
        except (RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_project_file(self, raw_path: str) -> None:
        candidate = (PROJECT_ROOT / urllib.parse.unquote(raw_path)).resolve()
        candidate.relative_to(PROJECT_ROOT.resolve())
        if not candidate.is_file():
            raise FileNotFoundError("Requested file does not exist.")
        stat = candidate.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(stat.st_size))
        self.end_headers()
        with candidate.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live parameter scan app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8770, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ParameterScanHandler)
    print(f"Parameter Scan Control: http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nParameter scan control stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
