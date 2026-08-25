"""Shared workload estimates and adaptive progress reporting for calibrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping


_TARGET_AXES = {"qubit", "qubit_pair"}
_REPETITION_AXES = {
    "n_runs",
    "shot",
    "shots",
    "shot_index",
    "sequence",
    "sequences",
    "nb_of_sequences",
    "random_sequence",
    "random_sequences",
}
_TIMING_SIGNATURE_FIELDS = (
    "reset_type",
    "use_state_discrimination",
    "transition",
    "operation",
    "pulse_shape",
    "echo",
    "ac_stark_correction",
)


@dataclass(frozen=True)
class RuntimeEstimate:
    """Approximate execution cost derived from sweep shape and saved runs."""

    sweep_points: int
    repetitions: int
    workload_units: int
    estimated_seconds: float | None = None
    historical_runs: int = 0
    source: str = "workload_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_duration(seconds: float | None) -> str:
    """Format seconds compactly for pre-run estimates and live ETAs."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result > 0 else default


def _axis_size(value: Any) -> int:
    size = getattr(value, "size", None)
    if size is not None:
        return _positive_int(size)
    try:
        return _positive_int(len(value))
    except (TypeError, AttributeError):
        return 1


def sweep_point_count(axes: Mapping[str, Any] | None) -> int:
    """Return the product of non-target sweep-axis lengths."""
    if not axes:
        return 1
    sizes = [
        _axis_size(axis)
        for name, axis in axes.items()
        if str(name).lower() not in _TARGET_AXES | _REPETITION_AXES
    ]
    return math.prod(sizes) if sizes else 1


def repetition_count(parameters: Any, progress_total: int | None) -> int:
    """Count outer repeats, including RB's nested shots-per-sequence case."""
    outer = _positive_int(progress_total)
    shots = _positive_int(getattr(parameters, "num_shots", None))
    if progress_total is not None and outer != shots:
        return outer * shots
    return outer


def workload_units(
    axes: Mapping[str, Any] | None,
    parameters: Any,
    progress_total: int | None,
) -> tuple[int, int, int]:
    """Return ``(sweep_points, repetitions, nominal acquisition points)``."""
    points = sweep_point_count(axes)
    repetitions = repetition_count(parameters, progress_total)
    return points, repetitions, points * repetitions


def _parameter_mapping(parameters: Any) -> dict[str, Any]:
    if parameters is None:
        return {}
    if isinstance(parameters, Mapping):
        return dict(parameters)
    if hasattr(parameters, "model_dump"):
        return dict(parameters.model_dump())
    if hasattr(parameters, "dict"):
        return dict(parameters.dict())
    try:
        values = vars(parameters)
    except TypeError:
        values = {
            name: getattr(parameters, name)
            for name in ("num_shots", "num_random_sequences", *_TIMING_SIGNATURE_FIELDS)
            if hasattr(parameters, name)
        }
    return {
        name: value
        for name, value in values.items()
        if not str(name).startswith("_")
    }


def _timing_signature(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: parameters[name]
        for name in _TIMING_SIGNATURE_FIELDS
        if name in parameters
    }


def _metadata_paths(output_root: Path, experiment_name: str) -> list[Path]:
    paths: list[Path] = []
    try:
        date_entries = list(os.scandir(output_root))
    except OSError:
        return paths
    for date_entry in date_entries:
        try:
            is_date_directory = date_entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if not is_date_directory:
            continue
        experiment_root = Path(date_entry.path) / experiment_name
        try:
            run_entries = list(os.scandir(experiment_root))
        except OSError:
            continue
        for run_entry in run_entries:
            try:
                is_run_directory = run_entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not is_run_directory:
                continue
            metadata_path = Path(run_entry.path) / "metadata.json"
            try:
                is_metadata_file = metadata_path.is_file()
            except OSError:
                continue
            if is_metadata_file:
                paths.append(metadata_path)
    return sorted(paths, key=lambda path: str(path), reverse=True)


def _historical_sweep_points(metadata: Mapping[str, Any]) -> int:
    sweep = metadata.get("sweep", {})
    if not isinstance(sweep, Mapping):
        return 1
    sizes: list[int] = []
    for name, details in sweep.items():
        if (
            str(name).lower() in _TARGET_AXES | _REPETITION_AXES
            or not isinstance(details, Mapping)
        ):
            continue
        shape = details.get("shape", [])
        # Saved xarray datasets can contain derived multidimensional coordinates.
        # Only 1-D coordinates correspond to independent sweep axes.
        if isinstance(shape, list) and len(shape) == 1:
            sizes.append(_positive_int(shape[0]))
    return math.prod(sizes) if sizes else 1


def _historical_workload(
    metadata: Mapping[str, Any], parameters: Mapping[str, Any]
) -> int:
    stored = metadata.get("runtime_estimate")
    if isinstance(stored, Mapping) and stored.get("workload_units") is not None:
        return _positive_int(stored["workload_units"])
    points = _historical_sweep_points(metadata)
    shots = _positive_int(parameters.get("num_shots"))
    sequences = parameters.get("num_random_sequences")
    repetitions = shots * _positive_int(sequences) if sequences is not None else shots
    return points * repetitions


def estimate_runtime(
    *,
    experiment_name: str,
    axes: Mapping[str, Any] | None,
    parameters: Any,
    progress_total: int | None,
    output_root: Path | str,
    history_limit: int = 5,
) -> RuntimeEstimate:
    """Scale recent comparable execution times by nominal acquisition count."""
    points, repetitions, current_workload = workload_units(
        axes, parameters, progress_total
    )
    current_parameters = _parameter_mapping(parameters)
    current_signature = _timing_signature(current_parameters)
    rates: list[float] = []

    for metadata_path in _metadata_paths(Path(output_root), experiment_name):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            parameters_path = metadata_path.with_name("parameters.json")
            try:
                has_parameters = parameters_path.is_file()
            except OSError:
                has_parameters = False
            old_parameters = (
                json.loads(parameters_path.read_text(encoding="utf-8"))
                if has_parameters
                else {}
            )
        except (OSError, json.JSONDecodeError):
            continue
        old_signature = _timing_signature(old_parameters)
        if any(
            name in old_signature and old_signature[name] != value
            for name, value in current_signature.items()
        ):
            continue
        duration = metadata.get("execution_duration_s", metadata.get("run_duration_s"))
        try:
            duration_s = float(duration)
        except (TypeError, ValueError, OverflowError):
            continue
        old_workload = _historical_workload(metadata, old_parameters)
        if duration_s <= 0 or old_workload <= 0:
            continue
        rates.append(duration_s / old_workload)
        if len(rates) >= history_limit:
            break

    if not rates:
        return RuntimeEstimate(points, repetitions, current_workload)
    return RuntimeEstimate(
        sweep_points=points,
        repetitions=repetitions,
        workload_units=current_workload,
        estimated_seconds=statistics.median(rates) * current_workload,
        historical_runs=len(rates),
        source="historical_workload_scaling",
    )


def progress_counter(
    iteration: Any,
    total: int,
    progress_bar: bool = True,
    percent: bool = True,
    start_time: float | None = None,
) -> None:
    """Drop-in progress reporter with an adaptive remaining-time estimate.

    Calibration programs save ``n`` at the beginning of each outer iteration.
    Therefore ``n`` is also the number of fully completed iterations and becomes a
    reliable rate estimate once it is at least one.
    """
    try:
        current = int(iteration)
    except (TypeError, ValueError, OverflowError):
        current = 0
    total = _positive_int(total)
    completed = min(max(current, 0), total)
    current_percent = completed / total * 100.0
    message = "Progress: "
    if progress_bar:
        filled = min(50, int(current_percent / 2))
        message += f"[{'#' * filled}{' ' * (50 - filled)}] "
    if percent:
        message += f"{current_percent:.1f}% ({completed}/{total} complete)"
    if start_time is not None:
        elapsed = max(0.0, time.time() - float(start_time))
        message += f" --> elapsed {format_duration(elapsed)}"
        if completed > 0:
            remaining = elapsed / completed * (total - completed)
            message += f", ETA {format_duration(remaining)}"
        else:
            message += ", ETA calibrating"
    print(message, end="\r")
