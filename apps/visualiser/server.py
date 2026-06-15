"""Dependency-free local server for browsing experiment and calibration data."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CALIBRATION_CODE_ROOT = PROJECT_ROOT / "calibrations"
CALIBRATION_RUN_ROOT = DATA_ROOT / "calibrations"
CALIBRATION_UPDATE_ROOT = DATA_ROOT / "calibration_updates"
PARAMETER_SCAN_ROOT = DATA_ROOT / "parameter_scans"

DATE_RE = re.compile(r"(?P<year>20\d{2})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})")
TIME_RE = re.compile(r"^\d{2}-\d{2}-\d{2}(?:-\d{6})?$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".yaml", ".yml", ".toml", ".ini"}
PREVIEW_LIMIT = 128_000
MAX_FILES = 600
MAX_PLOT_POINTS = 5000
MAX_PLOT_TRACES = 32
MAX_HEATMAP_AXIS = 400


def safe_iterdir(path: Path) -> tuple[list[Path], str | None]:
    try:
        return list(path.iterdir()), None
    except (OSError, PermissionError) as exc:
        return [], str(exc)


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except (OSError, PermissionError):
        return None


def iso_date_from_text(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime(
            int(match["year"]), int(match["month"]), int(match["day"])
        ).strftime("%Y-%m-%d")
    except ValueError:
        return None


def date_for_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        value = iso_date_from_text(part)
        if value:
            return value
    stat = safe_stat(path)
    return datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d") if stat else None


def display_time(path: Path) -> str:
    if TIME_RE.match(path.name):
        return path.name[:8].replace("-", ":")
    stat = safe_stat(path)
    return datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S") if stat else "--:--:--"


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def experiment_summary(path: Path, kind: str, experiment_type: str) -> dict[str, Any]:
    stat = safe_stat(path)
    qubits: list[str] = []
    profile_path = path / "profile" / "profile.json"
    profile_stat = safe_stat(profile_path)
    if profile_stat and profile_stat.st_size:
        profile, _ = read_json(profile_path)
        if isinstance(profile, dict) and isinstance(profile.get("active_qubits"), list):
            qubits = [str(qubit) for qubit in profile["active_qubits"]]
    return {
        "id": relative(path),
        "name": path.name if kind == "general" else experiment_type,
        "type": experiment_type,
        "kind": kind,
        "date": date_for_path(path),
        "time": display_time(path),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
        "path": relative(path),
        "status": "available" if stat else "unreadable",
        "qubits": qubits,
    }


def parameter_scan_experiments(date: str) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    date_root = PARAMETER_SCAN_ROOT / date
    scan_names, error = safe_iterdir(date_root)
    if error and date_root.exists():
        errors.append(f"Could not read parameter scan date directory: {error}")
    for scan_name in sorted((p for p in scan_names if p.is_dir()), key=lambda p: p.name):
        runs, error = safe_iterdir(scan_name)
        if error:
            errors.append(f"{relative(scan_name)}: {error}")
            continue
        for run in runs:
            if run.is_dir() and not run.name.startswith("."):
                results.append(experiment_summary(run, "parameter_scan", scan_name.name))
    return results, errors


def calibration_experiments(date: str) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    date_root = CALIBRATION_RUN_ROOT / date
    types, error = safe_iterdir(date_root)
    if error and date_root.exists():
        errors.append(f"Could not read calibration date directory: {error}")
    for type_dir in sorted((p for p in types if p.is_dir()), key=lambda p: p.name):
        runs, error = safe_iterdir(type_dir)
        if error:
            errors.append(f"{relative(type_dir)}: {error}")
            continue
        for run in runs:
            if run.is_dir() and not run.name.startswith("."):
                results.append(experiment_summary(run, "calibration", type_dir.name))
    return results, errors


def general_experiments(date: str) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    roots, error = safe_iterdir(DATA_ROOT)
    if error:
        return [], [f"Could not read data directory: {error}"]
    ignored = {"calibrations", "calibration_updates", "parameter_scans"}
    for category in (p for p in roots if p.is_dir() and p.name not in ignored):
        children, child_error = safe_iterdir(category)
        if child_error:
            errors.append(f"{relative(category)}: {child_error}")
            continue
        run_dirs = [p for p in children if p.is_dir() and not p.name.startswith(".")]
        if run_dirs:
            for run in run_dirs:
                if date_for_path(run) == date:
                    results.append(experiment_summary(run, "general", category.name))
        elif date_for_path(category) == date:
            results.append(experiment_summary(category, "general", category.name))
    return results, errors


def available_dates() -> list[str]:
    dates: set[str] = set()
    for root in (CALIBRATION_RUN_ROOT, CALIBRATION_UPDATE_ROOT, PARAMETER_SCAN_ROOT):
        children, _ = safe_iterdir(root)
        dates.update(filter(None, (iso_date_from_text(child.name) for child in children)))
    roots, _ = safe_iterdir(DATA_ROOT)
    for category in roots:
        if not category.is_dir() or category.name in {"calibrations", "calibration_updates", "parameter_scans"}:
            continue
        children, _ = safe_iterdir(category)
        candidates = [category, *children]
        dates.update(filter(None, (date_for_path(path) for path in candidates)))
    return sorted(dates, reverse=True)


def resolve_project_path(raw_path: str) -> Path:
    candidate = (PROJECT_ROOT / urllib.parse.unquote(raw_path)).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PermissionError("Path is outside the project") from exc
    return candidate


def collect_files(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if depth > 5 or len(files) >= MAX_FILES:
            return
        children, error = safe_iterdir(directory)
        if error:
            errors.append(f"{relative(directory)}: {error}")
            return
        for child in sorted(children, key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                visit(child, depth + 1)
            elif child.is_file():
                files.append(child)
                if len(files) >= MAX_FILES:
                    break

    visit(root, 0)
    if len(files) >= MAX_FILES:
        errors.append(f"File listing capped at {MAX_FILES} items.")
    return files, errors


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        if path.stat().st_size > PREVIEW_LIMIT:
            return None, f"JSON preview skipped because file exceeds {PREVIEW_LIMIT // 1000} KB."
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def preview_text(path: Path) -> tuple[str | None, str | None]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            value = file.read(PREVIEW_LIMIT)
        return value, None
    except OSError as exc:
        return None, str(exc)


def preview_csv(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                rows.append(row[:24])
                if index >= 30:
                    break
        return {"rows": rows, "truncated": len(rows) == 31}, None
    except (OSError, csv.Error) as exc:
        return None, str(exc)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = safe_stat(path)
    return {
        "name": path.name,
        "relative": path.relative_to(root).as_posix(),
        "project_path": relative(path),
        "extension": path.suffix.lower(),
        "size": stat.st_size if stat else None,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
        "url": "/api/file?path=" + urllib.parse.quote(relative(path)),
    }


def matching_calibration_assets(path: Path, experiment_type: str, date: str | None) -> dict[str, Any]:
    scripts = []
    for candidate in (
        CALIBRATION_CODE_ROOT / f"{experiment_type}.py",
        CALIBRATION_CODE_ROOT / f"{experiment_type.lower()}.py",
    ):
        if candidate.is_file():
            scripts.append(file_record(candidate, PROJECT_ROOT))
    updates: list[dict[str, Any]] = []
    update_root = CALIBRATION_UPDATE_ROOT / (date or "") / experiment_type
    if update_root.is_dir():
        run_dirs, _ = safe_iterdir(update_root)
        target_time = path.name[:8]
        for run in sorted((p for p in run_dirs if p.is_dir()), key=lambda p: p.name, reverse=True):
            if run.name[:8] == target_time or len(updates) < 3:
                files, errors = collect_files(run)
                updates.append(
                    {
                        "path": relative(run),
                        "time": display_time(run),
                        "files": [file_record(item, run) for item in files],
                        "errors": errors,
                    }
                )
            if len(updates) >= 4:
                break
    return {"scripts": scripts, "updates": updates}


def experiment_detail(raw_path: str) -> dict[str, Any]:
    path = resolve_project_path(raw_path)
    if not path.is_dir():
        raise FileNotFoundError("Experiment directory does not exist or cannot be read.")
    files, errors = collect_files(path)
    experiment_type = path.parent.name if path.parent.parent.name[:4].isdigit() else path.parent.name
    date = date_for_path(path)
    metadata: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    text: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    for item in files:
        record = file_record(item, path)
        if item.suffix.lower() in IMAGE_EXTENSIONS:
            figures.append(record)
        elif item.suffix.lower() == ".json":
            value, error = read_json(item)
            metadata.append({**record, "value": value, "error": error})
        elif item.suffix.lower() in {".csv", ".tsv"}:
            value, error = preview_csv(item)
            tables.append({**record, "value": value, "error": error})
        elif item.suffix.lower() in TEXT_EXTENSIONS:
            value, error = preview_text(item)
            text.append({**record, "value": value, "error": error})
        else:
            artifacts.append(record)

    return {
        "summary": experiment_summary(
            path,
            (
                "calibration"
                if CALIBRATION_RUN_ROOT.resolve() in path.resolve().parents
                else "parameter_scan"
                if PARAMETER_SCAN_ROOT.resolve() in path.resolve().parents
                else "general"
            ),
            experiment_type,
        ),
        "metadata": metadata,
        "tables": tables,
        "text": text,
        "figures": figures,
        "artifacts": artifacts,
        "calibrations": matching_calibration_assets(path, experiment_type, date),
        "errors": errors,
    }


def parameter_scan_data(raw_path: str) -> dict[str, Any]:
    path = resolve_project_path(raw_path)
    summary_path = path / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError("This run does not contain a parameter scan summary.csv.")

    def row_cycle(row: Mapping[str, str]) -> int:
        try:
            return int(row.get("cycle") or 0)
        except ValueError:
            return 0

    series: dict[tuple[str, str, str], dict[str, Any]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok" or not row.get("parameter"):
                continue
            try:
                value = float(row.get("value", ""))
            except ValueError:
                continue
            key = (row.get("experiment_name", ""), row.get("qubit", ""), row.get("parameter", ""))
            record = series.setdefault(
                key,
                {
                    "experiment_name": key[0],
                    "qubit": key[1],
                    "parameter": key[2],
                    "unit": row.get("unit", ""),
                    "points": [],
                },
            )
            record["points"].append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "cycle": row_cycle(row),
                    "value": value,
                    "success": row.get("success", ""),
                }
            )
    return {
        "path": relative(path),
        "series": sorted(series.values(), key=lambda item: (item["experiment_name"], item["qubit"], item["parameter"])),
    }


def _json_number(value: Any) -> float | None:
    number = float(abs(value)) if np.iscomplexobj(value) else float(value)
    return number if np.isfinite(number) else None


def _downsample(values: np.ndarray, indices: np.ndarray) -> list[float | None]:
    return [_json_number(value) for value in values.reshape(-1)[indices]]


def _matching_numeric_sweep(
    sweeps: dict[str, np.ndarray], size: int, excluded: set[str] | None = None
) -> tuple[str, np.ndarray] | None:
    excluded = excluded or set()
    for name, sweep in sweeps.items():
        if name in excluded or name == "qubit" or not np.issubdtype(sweep.dtype, np.number):
            continue
        if sweep.ndim == 1 and sweep.size == size:
            return name, sweep.astype(float)
    return None


def npz_plot_data(raw_path: str) -> dict[str, Any]:
    path = resolve_project_path(raw_path)
    if not path.is_dir():
        raise FileNotFoundError("Experiment directory does not exist or cannot be read.")
    sweep_path = path / "sweep.npz"
    results_path = path / "results.npz"
    if not safe_stat(sweep_path) or not safe_stat(results_path):
        raise FileNotFoundError("This experiment does not contain both sweep.npz and results.npz.")

    with np.load(sweep_path, allow_pickle=False) as sweep_file:
        sweeps = {name: np.asarray(sweep_file[name]) for name in sweep_file.files}
    with np.load(results_path, allow_pickle=False) as result_file:
        results = {name: np.asarray(result_file[name]) for name in result_file.files}

    qubits = [str(value) for value in sweeps.get("qubit", np.array([], dtype=str)).reshape(-1)]
    plot_results: list[dict[str, Any]] = []
    for result_name, array in results.items():
        if not np.issubdtype(array.dtype, np.number) or array.ndim == 0:
            continue
        point_count = int(array.shape[-1])
        x_name = "index"
        x_values = np.arange(point_count, dtype=float)
        x_sweep = _matching_numeric_sweep(sweeps, point_count)
        if x_sweep:
            x_name, x_values = x_sweep
        sample_indices = np.linspace(
            0, point_count - 1, min(point_count, MAX_PLOT_POINTS), dtype=int
        )
        leading_shape = array.shape[:-1]
        trace_indices = list(np.ndindex(leading_shape)) if leading_shape else [()]
        traces = []
        for index in trace_indices[:MAX_PLOT_TRACES]:
            values = np.asarray(array[index] if index else array).reshape(-1)
            label_parts = []
            if index and qubits and index[0] < len(qubits):
                label_parts.append(qubits[index[0]])
            if len(index) > 1 or (index and not label_parts):
                label_parts.append("slice " + ",".join(map(str, index)))
            traces.append(
                {
                    "label": " / ".join(label_parts) or result_name,
                    "y": _downsample(values, sample_indices),
                }
            )
        result_record = {
            "name": result_name,
            "shape": list(array.shape),
            "x_name": x_name,
            "x": _downsample(x_values, sample_indices),
            "traces": traces,
            "truncated_traces": len(trace_indices) > MAX_PLOT_TRACES,
            "downsampled": point_count > MAX_PLOT_POINTS,
            "heatmaps": [],
        }
        if array.ndim >= 2:
            y_size, x_size = int(array.shape[-2]), int(array.shape[-1])
            y_sweep = _matching_numeric_sweep(sweeps, y_size, {x_name})
            x_sweep_2d = _matching_numeric_sweep(sweeps, x_size, {y_sweep[0]} if y_sweep else set())
            if y_sweep and x_sweep_2d:
                y_name, y_values = y_sweep
                heat_x_name, heat_x_values = x_sweep_2d
                x_indices = np.linspace(0, x_size - 1, min(x_size, MAX_HEATMAP_AXIS), dtype=int)
                y_indices = np.linspace(0, y_size - 1, min(y_size, MAX_HEATMAP_AXIS), dtype=int)
                leading_shape_2d = array.shape[:-2]
                heatmap_indices = list(np.ndindex(leading_shape_2d)) if leading_shape_2d else [()]
                for index in heatmap_indices[:MAX_PLOT_TRACES]:
                    matrix = np.asarray(array[index] if index else array)
                    matrix = np.abs(matrix) if np.iscomplexobj(matrix) else matrix.astype(float)
                    label = qubits[index[0]] if index and qubits and index[0] < len(qubits) else (
                        "slice " + ",".join(map(str, index)) if index else result_name
                    )
                    result_record["heatmaps"].append(
                        {
                            "label": label,
                            "x_name": heat_x_name,
                            "y_name": y_name,
                            "x": _downsample(heat_x_values, x_indices),
                            "y": _downsample(y_values, y_indices),
                            "z": [
                                [_json_number(value) for value in matrix[row_index, x_indices]]
                                for row_index in y_indices
                            ],
                        }
                    )
        plot_results.append(result_record)
    return {
        "path": relative(path),
        "results": plot_results,
        "sweeps": [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sweeps.items()
        ],
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard assets and its read-only JSON API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT / "static"), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/dates":
                self.send_json({"dates": available_dates()})
                return
            if parsed.path == "/api/experiments":
                date = query.get("date", [""])[0]
                calibrations, calibration_errors = calibration_experiments(date)
                parameter_scans, parameter_scan_errors = parameter_scan_experiments(date)
                general, general_errors = general_experiments(date)
                experiments = sorted(
                    [*calibrations, *parameter_scans, *general],
                    key=lambda item: (item["time"], item["type"], item["path"]),
                    reverse=True,
                )
                self.send_json(
                    {
                        "experiments": experiments,
                        "errors": calibration_errors + parameter_scan_errors + general_errors,
                    }
                )
                return
            if parsed.path == "/api/experiment":
                self.send_json(experiment_detail(query.get("path", [""])[0]))
                return
            if parsed.path == "/api/npz-plot":
                self.send_json(npz_plot_data(query.get("path", [""])[0]))
                return
            if parsed.path == "/api/parameter-scan":
                self.send_json(parameter_scan_data(query.get("path", [""])[0]))
                return
            if parsed.path == "/api/file":
                self.serve_project_file(query.get("path", [""])[0])
                return
            super().do_GET()
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # Keep a malformed experiment from taking down the browser.
            self.send_json({"error": f"Could not process request: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_project_file(self, raw_path: str) -> None:
        path = resolve_project_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError("Requested file does not exist or cannot be read.")
        stat = path.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local data review dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Data Review Dashboard: http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
