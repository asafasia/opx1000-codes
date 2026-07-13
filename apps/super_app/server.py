"""Local home screen and process supervisor for the OPX1000 lab apps."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
DEFAULT_PORT = 8890
LAB_PYTHON = Path(r"C:\Users\owner\miniconda3\envs\opx1000_env\python.exe")
LOG_ROOT = PROJECT_ROOT / "data" / "app_logs" / "super_app"
MAX_RESTARTS = 4
MONITOR_INTERVAL_SECONDS = 2.0
MAX_MARKDOWN_BYTES = 2 * 1024 * 1024
WIKI_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class AppDefinition:
    id: str
    name: str
    eyebrow: str
    description: str
    port: int
    server_path: Path
    action: str
    title_marker: str


APPS = (
    AppDefinition(
        id="data-review",
        name="Data Review",
        eyebrow="Explore",
        description="Browse saved experiments, inspect plots, and review calibration results.",
        port=8892,
        server_path=PROJECT_ROOT / "apps" / "visualiser" / "server.py",
        action="Review data",
        title_marker="<title>Data Review Dashboard</title>",
    ),
    AppDefinition(
        id="lab-monitor",
        name="Lab Monitor",
        eyebrow="Observe",
        description="Watch active QOP jobs and live controller temperatures.",
        port=8895,
        server_path=PROJECT_ROOT / "apps" / "job_status" / "server.py",
        action="Open monitor",
        title_marker="<title>QOP Job Status</title>",
    ),
    AppDefinition(
        id="profile-studio",
        name="Profile Studio",
        eyebrow="Configure",
        description="Inspect and edit structured device profiles with guarded saves.",
        port=8893,
        server_path=PROJECT_ROOT / "apps" / "profile_studio" / "server.py",
        action="Edit profiles",
        title_marker="<title>Profile Studio</title>",
    ),
    AppDefinition(
        id="parameter-sweep",
        name="Parameter Sweep",
        eyebrow="Run",
        description="Configure long scans and follow stability, variance, and drift live.",
        port=8770,
        server_path=PROJECT_ROOT / "apps" / "parameter_scan" / "server.py",
        action="Open sweeps",
        title_marker="<title>Parameter Scan Control</title>",
    ),
)


def markdown_paths() -> list[Path]:
    """Return repository Markdown sources while skipping generated/private trees."""
    paths: list[Path] = []
    for root, directories, filenames in os.walk(PROJECT_ROOT):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in WIKI_EXCLUDED_DIRECTORIES and not directory.startswith(".")
        )
        root_path = Path(root)
        paths.extend(root_path / name for name in filenames if name.lower().endswith(".md"))
    return sorted(paths, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix().lower())


def markdown_metadata(path: Path) -> dict[str, Any]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    excerpt = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        heading = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        if heading:
            title = re.sub(r"[*_`]+", "", heading.group(1)).strip()
        for line in text.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith(("#", "```", "---", "<")):
                excerpt = re.sub(r"[*_`\[\]()>]", "", candidate)[:180]
                break
    except OSError:
        pass
    parts = Path(relative).parts
    section = parts[0] if len(parts) > 1 else "Repository root"
    return {
        "path": relative,
        "title": title,
        "section": section,
        "excerpt": excerpt,
        "size": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
    }


def wiki_index(query: str = "") -> dict[str, Any]:
    query_folded = query.strip().casefold()
    documents = []
    for path in markdown_paths():
        metadata = markdown_metadata(path)
        if query_folded:
            try:
                haystack = path.read_text(encoding="utf-8", errors="replace").casefold()
            except OSError:
                haystack = ""
            metadata_text = " ".join(
                str(metadata[key]) for key in ("path", "title", "section", "excerpt")
            ).casefold()
            if query_folded not in metadata_text and query_folded not in haystack:
                continue
        documents.append(metadata)
    return {"documents": documents, "count": len(documents), "query": query.strip()}


def resolve_wiki_path(raw_path: str, *, markdown_only: bool = True) -> Path:
    if not raw_path:
        raise FileNotFoundError("No wiki path was provided.")
    candidate = (PROJECT_ROOT / raw_path.replace("/", os.sep)).resolve()
    try:
        relative = candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("Wiki paths must stay inside the repository.") from exc
    if any(part in WIKI_EXCLUDED_DIRECTORIES or part.startswith(".") for part in relative.parts):
        raise PermissionError("That repository path is not exposed by the wiki.")
    if markdown_only and candidate.suffix.lower() != ".md":
        raise PermissionError("The wiki can only open Markdown documents.")
    if not candidate.is_file():
        raise FileNotFoundError("Wiki document was not found.")
    return candidate


def read_wiki_document(raw_path: str) -> dict[str, Any]:
    path = resolve_wiki_path(raw_path)
    if path.stat().st_size > MAX_MARKDOWN_BYTES:
        raise ValueError("Markdown document is too large to display.")
    return {**markdown_metadata(path), "content": path.read_text(encoding="utf-8", errors="replace")}


def port_is_open(host: str, port: int, timeout: float = 0.15) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_app(host: str, app: AppDefinition, timeout: float = 0.75) -> tuple[bool, str | None]:
    """Verify that a port serves the expected app, not just any listener."""
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        request = urllib.request.Request(
            f"http://{probe_host}:{app.port}/",
            headers={"User-Agent": "OPX1000-Super-App/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(128_000).decode("utf-8", errors="replace")
            if response.status != HTTPStatus.OK:
                return False, f"Health check returned HTTP {response.status}."
            if app.title_marker not in body:
                return False, f"Port {app.port} is serving a different application."
        return True, None
    except (OSError, urllib.error.URLError) as exc:
        return False, str(exc)


class AppManager:
    def __init__(self, host: str, *, launch_apps: bool = True) -> None:
        self.host = host
        self.launch_apps = launch_apps
        self.processes: dict[str, subprocess.Popen[Any]] = {}
        self.log_files: dict[str, Any] = {}
        self.restart_counts: dict[str, int] = {}
        self.next_restart_at: dict[str, float] = {}
        self.last_errors: dict[str, str] = {}
        self.started_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def start_all(self) -> None:
        if not self.launch_apps:
            return
        for app in APPS:
            self._start(app)
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            name="super-app-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _start(self, app: AppDefinition) -> None:
        healthy, error = probe_app(self.host, app)
        if healthy:
            self.last_errors.pop(app.id, None)
            return
        if port_is_open(self.host, app.port):
            self.last_errors[app.id] = error or f"Port {app.port} is already in use."
            return

        python = LAB_PYTHON if LAB_PYTHON.is_file() else Path(sys.executable)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{app.id}.log"
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {app.name}\n")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [str(python), str(app.server_path), "--host", self.host, "--port", str(app.port)],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as exc:
            log_file.close()
            self.last_errors[app.id] = f"Could not start: {exc}"
            return
        with self._lock:
            old_log = self.log_files.pop(app.id, None)
            if old_log is not None:
                old_log.close()
            self.processes[app.id] = process
            self.log_files[app.id] = log_file
            self.started_at[app.id] = time.time()
            self.last_errors[app.id] = "Starting…"

    def _monitor(self) -> None:
        while not self._stop_event.wait(MONITOR_INTERVAL_SECONDS):
            for app in APPS:
                with self._lock:
                    process = self.processes.get(app.id)
                if process is None or process.poll() is None:
                    continue
                healthy, _ = probe_app(self.host, app)
                if healthy:
                    continue
                count = self.restart_counts.get(app.id, 0)
                if count >= MAX_RESTARTS:
                    self.last_errors[app.id] = (
                        f"Exited with code {process.returncode}; restart limit reached. "
                        f"See {self.log_path(app).as_posix()}."
                    )
                    continue
                now = time.time()
                if now < self.next_restart_at.get(app.id, 0):
                    continue
                count += 1
                self.restart_counts[app.id] = count
                self.next_restart_at[app.id] = now + min(30, 2**count)
                self.last_errors[app.id] = f"Exited with code {process.returncode}; restarting ({count}/{MAX_RESTARTS})."
                self._start(app)

    @staticmethod
    def log_path(app: AppDefinition) -> Path:
        return LOG_ROOT / f"{app.id}.log"

    def status(self, app: AppDefinition) -> dict[str, Any]:
        with self._lock:
            process = self.processes.get(app.id)
        running, probe_error = probe_app(self.host, app)
        process_alive = process is not None and process.poll() is None
        if running:
            self.last_errors.pop(app.id, None)
        elif process_alive and time.time() - self.started_at.get(app.id, 0) < 20:
            self.last_errors[app.id] = "Starting…"
        elif probe_error and not self.last_errors.get(app.id):
            self.last_errors[app.id] = probe_error
        return {
            "id": app.id,
            "name": app.name,
            "eyebrow": app.eyebrow,
            "description": app.description,
            "port": app.port,
            "action": app.action,
            "running": running,
            "managed": process_alive,
            "exit_code": process.poll() if process is not None and not running else None,
            "error": None if running else self.last_errors.get(app.id),
            "restart_count": self.restart_counts.get(app.id, 0),
            "log_path": self.log_path(app).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        }

    def payload(self) -> dict[str, Any]:
        return {"apps": [self.status(app) for app in APPS], "checked_at": time.time()}

    def stop_all(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=MONITOR_INTERVAL_SECONDS + 1)
        with self._lock:
            processes = list(self.processes.values())
            self.processes.clear()
        for process in processes:
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.send_signal(signal.SIGINT)
                except (OSError, ValueError):
                    process.terminate()
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.terminate()
        for log_file in self.log_files.values():
            log_file.close()
        self.log_files.clear()


class SuperAppServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], manager: AppManager) -> None:
        self.manager = manager
        super().__init__(address, SuperAppHandler)


class SuperAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT / "static"), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        parameters = urllib.parse.parse_qs(parsed.query)
        if path == "/api/apps":
            server = self.server
            assert isinstance(server, SuperAppServer)
            self.send_json(server.manager.payload())
            return
        if path == "/api/wiki":
            self.send_json(wiki_index(parameters.get("q", [""])[-1]))
            return
        if path == "/api/wiki/file":
            try:
                self.send_json(read_wiki_document(parameters.get("path", [""])[-1]))
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (OSError, ValueError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/wiki/asset":
            try:
                asset = resolve_wiki_path(parameters.get("path", [""])[-1], markdown_only=False)
                if asset.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
                    raise PermissionError("Only wiki image assets can be displayed.")
                self.serve_file(asset)
            except PermissionError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path in {"/assets/grouplogo.png", "/assets/Q.png"}:
            self.serve_logo(Path(path).name)
            return
        super().do_GET()

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_logo(self, name: str) -> None:
        path = PROJECT_ROOT / "apps" / "visualiser" / "static" / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path) -> None:
        stat = path.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OPX1000 lab app home screen.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Show links without starting the four linked app servers.",
    )
    args = parser.parse_args()
    manager = AppManager(args.host, launch_apps=not args.no_launch)
    server = SuperAppServer((args.host, args.port), manager)
    print(f"OPX1000 Lab Home: http://{args.host}:{args.port}")
    print("Linked apps: Data Review, Lab Monitor, Profile Studio, Parameter Sweep")
    try:
        # Bind the hub first so a port conflict cannot orphan newly launched apps.
        manager.start_all()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOPX1000 Lab Home stopped.")
    finally:
        server.server_close()
        manager.stop_all()


if __name__ == "__main__":
    main()
