"""Dependency-free local editor for repository device profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
PROFILES_ROOT = PROJECT_ROOT / "profiles"
EDITABLE_FILES = {
    "profile": "profile.json",
    "qubits": "qubits.json",
    "pulses": "pulses.json",
    "connectivity": "connectivity.json",
}
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def profile_directory(name: str) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise PermissionError("Invalid profile name.")
    path = (PROFILES_ROOT / name).resolve()
    try:
        path.relative_to(PROFILES_ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("Profile is outside profiles/.") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")
    return path


def editable_path(profile: str, section: str) -> Path:
    if section not in EDITABLE_FILES:
        raise PermissionError("That profile section is not editable.")
    path = profile_directory(profile) / EDITABLE_FILES[section]
    if not path.is_file():
        raise FileNotFoundError(f"{EDITABLE_FILES[section]} does not exist.")
    return path


def list_profiles() -> list[str]:
    if not PROFILES_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in PROFILES_ROOT.iterdir()
        if path.is_dir() and all((path / filename).is_file() for filename in EDITABLE_FILES.values())
    )


def read_section(profile: str, section: str) -> dict[str, Any]:
    path = editable_path(profile, section)
    text = path.read_text(encoding="utf-8")
    return {
        "profile": profile,
        "section": section,
        "filename": path.name,
        "data": json.loads(text),
        "digest": digest(text),
    }


def write_section(profile: str, section: str, data: Any, expected_digest: str) -> dict[str, Any]:
    path = editable_path(profile, section)
    current_text = path.read_text(encoding="utf-8")
    if digest(current_text) != expected_digest:
        raise FileExistsError("The file changed on disk after it was loaded. Reload before saving.")

    rendered = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    json.loads(rendered)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            file.write(rendered)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"profile": profile, "section": section, "digest": digest(rendered)}


class ProfileStudioHandler(SimpleHTTPRequestHandler):
    """Serve Profile Studio and its narrowly scoped write API."""

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

    def request_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is empty or too large.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, _, query = self.path.partition("?")
            parameters = {
                key: values[-1]
                for key, values in urllib.parse.parse_qs(query).items()
            }
            if path == "/api/profiles":
                self.send_json({"profiles": list_profiles(), "sections": list(EDITABLE_FILES)})
                return
            if path == "/api/section":
                self.send_json(read_section(parameters.get("profile", ""), parameters.get("section", "")))
                return
            if path == "/assets/grouplogo.png":
                self.serve_logo("grouplogo.png")
                return
            if path == "/assets/Q.png":
                self.serve_logo("Q.png")
                return
            super().do_GET()
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            self.send_json({"error": f"Invalid profile data: {exc}"}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self.send_json({"error": f"Could not read profile: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            if self.path != "/api/section":
                self.send_json({"error": "Unknown API endpoint."}, HTTPStatus.NOT_FOUND)
                return
            request = self.request_json()
            result = write_section(
                str(request.get("profile", "")),
                str(request.get("section", "")),
                request.get("data"),
                str(request.get("digest", "")),
            )
            self.send_json(result)
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except FileExistsError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (json.JSONDecodeError, UnicodeError, ValueError, TypeError) as exc:
            self.send_json({"error": f"Invalid profile data: {exc}"}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self.send_json({"error": f"Could not save profile: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_logo(self, name: str) -> None:
        path = PROJECT_ROOT / "apps" / "visualiser" / "static" / name
        if not path.is_file():
            raise FileNotFoundError(f"Shared logo {name} was not found.")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local device profile editor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ProfileStudioHandler)
    print(f"Profile Studio: http://{args.host}:{args.port}")
    print(f"Profiles root: {PROFILES_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProfile Studio stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
