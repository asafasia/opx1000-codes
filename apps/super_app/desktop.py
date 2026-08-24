"""Native Windows host for the connected OPX1000 lab applications."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
import urllib.error
import urllib.request
from http import HTTPStatus
from types import ModuleType
from typing import Any


_NULL_STREAMS: list[Any] = []
for stream_name in ("stdout", "stderr"):
    if getattr(sys, stream_name) is None:
        stream = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        _NULL_STREAMS.append(stream)
        setattr(sys, stream_name, stream)


try:  # Source-tree import.
    from apps.super_app.server import (  # type: ignore[no-redef]  # noqa: E402
        AppManager,
        DEFAULT_PORT,
        LOG_ROOT,
        SuperAppServer,
        port_is_open,
    )
except ModuleNotFoundError:  # Standalone project / frozen executable import.
    from server import (  # type: ignore[no-redef]  # noqa: E402
        AppManager,
        DEFAULT_PORT,
        LOG_ROOT,
        SuperAppServer,
        port_is_open,
    )


WINDOW_TITLE = "OPX1000 Quantum Coherence Lab"
WINDOW_WIDTH = 1480
WINDOW_HEIGHT = 940
WINDOW_MIN_WIDTH = 980
WINDOW_MIN_HEIGHT = 680
HUB_TITLE_MARKER = "<title>OPX1000 Quantum Coherence Lab</title>"


def probe_existing_hub(host: str, port: int, timeout: float = 0.75) -> tuple[bool, str | None]:
    """Verify that an existing listener is this repository's lab hub."""
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        request = urllib.request.Request(
            f"http://{probe_host}:{port}/",
            headers={"User-Agent": "OPX1000-Desktop/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(128_000).decode("utf-8", errors="replace")
            if response.status != HTTPStatus.OK or HUB_TITLE_MARKER not in body:
                return False, f"Port {port} is serving a different application."
        return True, None
    except (OSError, urllib.error.URLError) as exc:
        return False, str(exc)


class DesktopRuntime:
    """Own the local HTTP services used by the native desktop window."""

    def __init__(self, host: str, port: int, *, launch_apps: bool = True) -> None:
        self.host = host
        self.port = port
        self.manager = AppManager(host, launch_apps=launch_apps)
        self.server: SuperAppServer | None = None
        self.server_thread: threading.Thread | None = None
        self.services_thread: threading.Thread | None = None
        self.using_existing_hub = False
        self._stop_lock = threading.Lock()
        self._stopped = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?desktop=1"

    def start(self) -> None:
        """Serve the hub immediately and launch linked services in the background."""
        if self.server is not None:
            raise RuntimeError("The desktop runtime is already running.")

        healthy, probe_error = probe_existing_hub(self.host, self.port)
        if healthy:
            self.using_existing_hub = True
            return
        if port_is_open(self.host, self.port):
            raise RuntimeError(probe_error or f"Port {self.port} is already in use.")

        server = SuperAppServer((self.host, self.port), self.manager)
        self.server = server
        try:
            thread = threading.Thread(
                target=server.serve_forever,
                name="opx1000-desktop-server",
                daemon=True,
            )
            self.server_thread = thread
            thread.start()
            services_thread = threading.Thread(
                target=self.manager.start_all,
                name="quantum-coherence-lab-services",
                daemon=True,
            )
            self.services_thread = services_thread
            services_thread.start()
        except BaseException:
            server.server_close()
            self.server = None
            self.manager.stop_all()
            raise

    def stop(self, *_args: Any) -> None:
        """Stop the hub and only the child services owned by this runtime."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        server = self.server
        thread = self.server_thread
        services_thread = self.services_thread
        self.manager.stop_all()
        if services_thread is not None and services_thread.is_alive():
            services_thread.join(timeout=5)
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
                thread.join(timeout=5)
            server.server_close()


def load_webview() -> ModuleType:
    """Load the optional native UI dependency with an actionable error."""
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "The OPX1000 desktop runtime is not installed. Run "
            "'C:\\Users\\owner\\miniconda3\\envs\\opx1000_env\\python.exe "
            "-m pip install pywebview', then open the app again."
        ) from exc
    return webview


def show_startup_error(message: str) -> None:
    """Show errors even when launched through pythonw without a console."""
    try:
        from tkinter import Tk, messagebox

        root = Tk()
        root.withdraw()
        messagebox.showerror(f"{WINDOW_TITLE} could not start", message)
        root.destroy()
    except Exception:
        print(message, file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the connected OPX1000 lab desktop app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Open the desktop home without starting the linked services.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable WebView developer tools.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = DesktopRuntime(args.host, args.port, launch_apps=not args.no_launch)
    window_closed = threading.Event()

    def mark_window_closed(*_args: Any) -> None:
        window_closed.set()

    try:
        webview = load_webview()
        runtime.start()
        window = webview.create_window(
            WINDOW_TITLE,
            runtime.url,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
            text_select=True,
        )
        window.events.closed += mark_window_closed
        webview.start(gui="edgechromium", debug=args.debug)
        # Some frozen WebView hosts return from start() while the native window's
        # .NET message loop is still active. Keep the local services alive until
        # that window has actually closed.
        window_closed.wait()
        return 0
    except Exception as exc:
        try:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            (LOG_ROOT / "desktop-startup-error.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
        except OSError:
            pass
        show_startup_error(str(exc))
        return 1
    finally:
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())
