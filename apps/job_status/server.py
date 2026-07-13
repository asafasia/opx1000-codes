"""Dependency-free local server for live QOP job status."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrations.job_status import query_profile_qop_status
from profiles import load_profile
from temperature_monitor.temperature_monitor import TemperatureMonitor


DEFAULT_PORT = 8895
LAB_PYTHON = Path(r"C:\Users\owner\miniconda3\envs\opx1000_env\python.exe")
TEMPERATURE_LOG_ROOT = PROJECT_ROOT / "data" / "temperature_logs"
TEMPERATURE_RETENTION_DAYS = 7
TEMPERATURE_POLL_INTERVAL_SECONDS = 5.0
TEMPERATURE_HISTORY_POINTS = 240
TEMPERATURE_HISTORY_PAYLOAD_POINTS = 1200
MIN_TEMPERATURE_WINDOW_SECONDS = 60

temperature_sampler: "LiveTemperatureSampler | None" = None


class JobStatusServer(ThreadingHTTPServer):
    allow_reuse_address = False


def maybe_reexec_lab_python() -> None:
    """Keep the dashboard in the OPX lab environment when launched directly."""
    if os.environ.get("JOB_STATUS_REEXECED") == "1":
        return
    if not LAB_PYTHON.is_file():
        return
    if Path(sys.executable).resolve() == LAB_PYTHON.resolve():
        return

    environment = dict(os.environ)
    environment["JOB_STATUS_REEXECED"] = "1"
    raise SystemExit(
        subprocess.call([str(LAB_PYTHON), *sys.argv], cwd=PROJECT_ROOT, env=environment)
    )


def jobs_payload(
    *,
    profile_name: str | None = None,
    qubit: str | None = None,
    all_jobs: bool = False,
) -> dict[str, Any]:
    """Return the serializable API payload shown by the dashboard."""
    status = query_profile_qop_status(
        profile_name=profile_name or None,
        qubit=qubit or None,
        active_only=not all_jobs,
    )
    payload = status.to_dict()
    payload.update(
        {
            "profile": profile_name or "default",
            "qubit": qubit or None,
            "active_only": not all_jobs,
            "polled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return payload


class LiveTemperatureSampler:
    """Continuously sample controller temperatures and append a live CSV log."""

    def __init__(
        self,
        *,
        poll_interval: float = TEMPERATURE_POLL_INTERVAL_SECONDS,
        output_root: Path = TEMPERATURE_LOG_ROOT,
        retention_days: int = TEMPERATURE_RETENTION_DAYS,
    ) -> None:
        self.poll_interval = poll_interval
        self.output_root = Path(output_root)
        self.retention_days = retention_days
        self.monitor: TemperatureMonitor | None = None
        self.started_at: datetime | None = None
        self.latest_temperatures: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self.error: str | None = None
        self.connected = False
        self.output_dir: Path | None = None
        self.csv_path: Path | None = None
        self.temperature_keys: list[Any] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prune = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="live-temperature-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval * 2))
        if self.monitor is not None:
            self.monitor._safe_save()

    def payload(self, *, window_seconds: int | None = None) -> dict[str, Any]:
        with self._lock:
            payload = {
                "connected": self.connected,
                "error": self.error,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "latest": dict(self.latest_temperatures),
                "history": [],
                "retention_days": self.retention_days,
                "poll_interval_seconds": self.poll_interval,
            }
            memory_history = list(self.history)
        payload["history"] = self._history_for_window(memory_history, window_seconds)
        return payload

    def _connect(self) -> None:
        self._prune_old_logs()
        network = load_profile("main")["connectivity"]["network"]
        monitor = TemperatureMonitor(
            controller_name="con1",
            poll_interval=self.poll_interval,
            max_points=TEMPERATURE_HISTORY_POINTS,
            fem_ids=(6, 2),
            include_chassis_and_crps0=True,
            save_dir=self.output_root / "live",
            register_exit_handlers=False,
            qop_host=network["host"],
            qop_cluster_name=network["cluster_name"],
        )
        monitor._initialize_history()
        self.monitor = monitor
        self.started_at = monitor.started_at.replace(tzinfo=timezone.utc)
        self.output_dir = monitor.output_dir
        self.csv_path = self.output_dir / "temperature_log_live.csv"
        self.temperature_keys = list(monitor.temperature_keys)
        self.error = None
        self.connected = True

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.monitor is None:
                    self._connect()
                self._sample()
            except Exception as exc:
                with self._lock:
                    self.connected = False
                    self.error = str(exc)
                self.monitor = None
            self._stop_event.wait(self.poll_interval)

    def _sample(self) -> None:
        if self.monitor is None:
            return
        self._prune_old_logs()
        self.monitor.sample_once()
        elapsed = self.monitor.elapsed_seconds[-1]
        latest = {
            str(key): float(self.monitor.temperature_history[key][-1])
            for key in self.monitor.temperature_keys
        }
        sample = {
            "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(elapsed),
            "temperatures": latest,
        }
        with self._lock:
            self.connected = True
            self.error = None
            self.latest_temperatures = latest
            self.history.append(sample)
            self.history = self.history[-TEMPERATURE_HISTORY_POINTS:]
        self._append_csv(sample)

    def _append_csv(self, sample: dict[str, Any]) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["wall_time_utc", "elapsed_seconds", *map(str, self.temperature_keys)])
            writer.writerow(
                [
                    sample["time"],
                    sample["elapsed_seconds"],
                    *[
                        sample["temperatures"].get(str(key), "")
                        for key in self.temperature_keys
                    ],
                ]
            )

    def _history_for_window(
        self,
        memory_history: list[dict[str, Any]],
        window_seconds: int | None,
    ) -> list[dict[str, Any]]:
        if not window_seconds:
            return memory_history
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        samples = self._read_logged_history(cutoff)
        if not samples:
            samples = [
                sample
                for sample in memory_history
                if parse_iso_datetime(sample.get("time")) >= cutoff
            ]
        return downsample_samples(samples, TEMPERATURE_HISTORY_PAYLOAD_POINTS)

    def _read_logged_history(self, cutoff: datetime) -> list[dict[str, Any]]:
        live_root = self.output_root / "live"
        if not live_root.is_dir():
            return []

        samples: list[dict[str, Any]] = []
        stale_cutoff = cutoff - timedelta(minutes=10)
        for csv_path in sorted(live_root.glob("*/temperature_log_live.csv")):
            try:
                if datetime.fromtimestamp(csv_path.stat().st_mtime, timezone.utc) < stale_cutoff:
                    continue
            except OSError:
                continue
            try:
                samples.extend(read_temperature_csv(csv_path, cutoff))
            except (OSError, csv.Error, ValueError):
                continue
        samples.sort(key=lambda sample: parse_iso_datetime(sample.get("time")))
        return samples

    def _prune_old_logs(self) -> None:
        now = time.time()
        if now - self._last_prune < 3600:
            return
        self._last_prune = now
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        for parent in (self.output_root, self.output_root / "live"):
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if not path.is_dir():
                    continue
                try:
                    modified = datetime.fromtimestamp(path.stat().st_mtime)
                except OSError:
                    continue
                if modified < cutoff:
                    shutil.rmtree(path, ignore_errors=True)


def parse_iso_datetime(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_temperature_csv(csv_path: Path, cutoff: datetime) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return samples
        temperature_keys = [
            key
            for key in reader.fieldnames
            if key not in {"wall_time_utc", "elapsed_seconds"}
        ]
        for row in reader:
            timestamp = parse_iso_datetime(row.get("wall_time_utc"))
            if timestamp < cutoff:
                continue
            temperatures = {}
            for key in temperature_keys:
                raw_value = row.get(key)
                if raw_value in {None, ""}:
                    continue
                temperatures[key] = float(raw_value)
            if not temperatures:
                continue
            try:
                elapsed_seconds = float(row.get("elapsed_seconds") or 0)
            except ValueError:
                elapsed_seconds = 0.0
            samples.append(
                {
                    "time": timestamp.isoformat(),
                    "elapsed_seconds": elapsed_seconds,
                    "temperatures": temperatures,
                }
            )
    return samples


def downsample_samples(
    samples: list[dict[str, Any]],
    max_points: int,
) -> list[dict[str, Any]]:
    if len(samples) <= max_points:
        return samples
    step = math.ceil(len(samples) / max_points)
    reduced = samples[::step]
    if reduced[-1] is not samples[-1]:
        reduced.append(samples[-1])
    return reduced


def parse_window_seconds(parameters: dict[str, str]) -> int | None:
    raw_value = parameters.get("window_seconds")
    if not raw_value:
        return 15 * 60
    try:
        value = int(raw_value)
    except ValueError:
        return 15 * 60
    max_seconds = TEMPERATURE_RETENTION_DAYS * 24 * 60 * 60
    return min(max(value, MIN_TEMPERATURE_WINDOW_SECONDS), max_seconds)


def temperature_payload(*, window_seconds: int | None = None) -> dict[str, Any]:
    if temperature_sampler is None:
        return {
            "connected": False,
            "error": "Temperature sampler is not running.",
            "latest": {},
            "history": [],
            "window_seconds": window_seconds,
        }
    payload = temperature_sampler.payload(window_seconds=window_seconds)
    payload["window_seconds"] = window_seconds
    return payload


def list_temperature_scans(limit: int = 10) -> list[dict[str, Any]]:
    roots = [TEMPERATURE_LOG_ROOT, TEMPERATURE_LOG_ROOT / "live"]
    scans: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        scans.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and not (root == TEMPERATURE_LOG_ROOT and path.name == "live")
        )
    unique = sorted(set(scans), key=lambda path: path.stat().st_mtime, reverse=True)
    results = []
    for path in unique[:limit]:
        results.append(
            {
                "name": path.name,
                "path": str(path),
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "has_live_csv": (path / "temperature_log_live.csv").is_file(),
                "has_final_csv": (path / "temperature_log.csv").is_file(),
            }
        )
    return results


def status_payload(
    *,
    profile_name: str | None = None,
    qubit: str | None = None,
    all_jobs: bool = False,
    window_seconds: int | None = None,
) -> dict[str, Any]:
    temperature = temperature_payload(window_seconds=window_seconds)
    try:
        jobs = jobs_payload(profile_name=profile_name, qubit=qubit, all_jobs=all_jobs)
        jobs_error = None
    except Exception as exc:
        jobs = {
            "open_qms": [],
            "has_active_jobs": False,
            "jobs": [],
            "profile": profile_name or "default",
            "qubit": qubit or None,
            "active_only": not all_jobs,
            "polled_at": datetime.now(timezone.utc).isoformat(),
        }
        jobs_error = str(exc)
    qpu_live = bool(temperature.get("connected"))
    return {
        "qpu_live": qpu_live,
        "jobs": jobs,
        "jobs_error": jobs_error,
        "temperature": temperature,
        "polled_at": datetime.now(timezone.utc).isoformat(),
    }


class JobStatusHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard and its read-only status API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT / "static"), **kwargs)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path not in {"/api/jobs", "/api/temperature", "/api/status"}:
            super().do_GET()
            return

        parameters = {
            key: values[-1]
            for key, values in urllib.parse.parse_qs(query).items()
        }
        all_jobs = parameters.get("all", "").lower() in {"1", "true", "yes", "on"}
        window_seconds = parse_window_seconds(parameters)
        try:
            if path == "/api/jobs":
                self.send_json(
                    jobs_payload(
                        profile_name=parameters.get("profile") or None,
                        qubit=parameters.get("qubit") or None,
                        all_jobs=all_jobs,
                    )
                )
                return
            if path == "/api/temperature":
                self.send_json(temperature_payload(window_seconds=window_seconds))
                return
            self.send_json(
                status_payload(
                    profile_name=parameters.get("profile") or None,
                    qubit=parameters.get("qubit") or None,
                    all_jobs=all_jobs,
                    window_seconds=window_seconds,
                )
            )
        except Exception as exc:  # Keep the UI alive when QOP is unreachable.
            self.send_json(
                {
                    "error": str(exc),
                    "profile": parameters.get("profile") or "default",
                    "qubit": parameters.get("qubit") or None,
                    "active_only": not all_jobs,
                    "polled_at": datetime.now(timezone.utc).isoformat(),
                },
                HTTPStatus.BAD_GATEWAY,
            )


def main() -> None:
    global temperature_sampler
    maybe_reexec_lab_python()
    parser = argparse.ArgumentParser(description="Run the local QOP job dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--temperature-interval",
        default=TEMPERATURE_POLL_INTERVAL_SECONDS,
        type=float,
        help="Seconds between controller temperature samples.",
    )
    args = parser.parse_args()
    temperature_sampler = LiveTemperatureSampler(poll_interval=args.temperature_interval)
    temperature_sampler.start()
    server = JobStatusServer((args.host, args.port), JobStatusHandler)
    print(f"QOP Job Status: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nQOP Job Status stopped.")
    finally:
        if temperature_sampler is not None:
            temperature_sampler.stop()
        server.server_close()


if __name__ == "__main__":
    main()
