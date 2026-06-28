"""Save raw calibration results with a snapshot of the device profile."""

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from profiles import Profile, current_profile_name
from utils.plotting_settings import add_calibration_timestamp


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "calibrations"
DEFAULT_PROFILES_ROOT = REPOSITORY_ROOT / "profiles"
_VALID_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_name(name: str, label: str) -> str:
    if not name or not _VALID_NAME.fullmatch(name):
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_' or '-'")
    return name


def _validate_array_name(name: str) -> str:
    if not name or "/" in name or "\\" in name:
        raise ValueError("array name must be non-empty and cannot contain path separators")
    return name


def _profile_name(profile: str | Profile | None) -> str:
    return profile.name if isinstance(profile, Profile) else profile or current_profile_name()


def _as_arrays(values: Mapping[str, Any] | Any, default_name: str) -> dict[str, np.ndarray]:
    items = values.items() if isinstance(values, Mapping) else [(default_name, values)]
    arrays: dict[str, np.ndarray] = {}

    for name, value in items:
        name = _validate_array_name(str(name))
        array = np.asarray(value)
        if array.dtype.hasobject:
            if all(isinstance(item, str) for item in array.flat):
                array = array.astype(str)
            else:
                raise ValueError(f"Array {name!r} has object dtype and cannot be saved safely")
        arrays[name] = array

    if not arrays:
        raise ValueError("At least one array must be provided")
    return arrays


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }


def _to_jsonable(value: Any) -> Any:
    """Convert calibration parameters to JSON-safe built-in types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _parameter_metadata(parameters: Any | None) -> dict[str, Any] | None:
    if parameters is None:
        return None
    return _to_jsonable(parameters)


def _profile_snapshot_ignore(profile_source: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == profile_source and "kernels" in names:
            return {"kernels"}
        return set()

    return ignore


class CalibrationSaver:
    """Save calibration arrays under a date, experiment name, and run time."""

    def __init__(
        self,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        profiles_root: Path | str = DEFAULT_PROFILES_ROOT,
    ) -> None:
        self.output_root = Path(output_root)
        self.profiles_root = Path(profiles_root)

    def save(
        self,
        experiment_name: str,
        sweep: Mapping[str, Any] | Any,
        results: Mapping[str, Any] | Any,
        profile_name: str | Profile | None = None,
        parameters: Any | None = None,
        now: datetime | None = None,
    ) -> Path:
        """Save arrays and return the newly created run directory."""
        experiment_name = _validate_name(experiment_name, "experiment_name")
        profile_name = _validate_name(_profile_name(profile_name), "profile_name")
        profile_source = self.profiles_root / profile_name
        if not profile_source.is_dir():
            raise FileNotFoundError(f"Profile directory does not exist: {profile_source}")

        sweep_arrays = _as_arrays(sweep, "sweep")
        result_arrays = _as_arrays(results, "results")
        timestamp = now or datetime.now().astimezone()
        experiment_root = self.output_root / timestamp.strftime("%Y-%m-%d") / experiment_name
        run_name = timestamp.strftime("%H-%M-%S-%f")
        run_directory = experiment_root / run_name
        experiment_root.mkdir(parents=True, exist_ok=True)

        temporary_directory = Path(tempfile.mkdtemp(prefix=f".{run_name}-", dir=experiment_root))
        try:
            np.savez_compressed(temporary_directory / "sweep.npz", **sweep_arrays)
            np.savez_compressed(temporary_directory / "results.npz", **result_arrays)
            shutil.copytree(
                profile_source,
                temporary_directory / "profile",
                ignore=_profile_snapshot_ignore(profile_source),
            )
            serialized_parameters = _parameter_metadata(parameters)
            if serialized_parameters is not None:
                with (temporary_directory / "parameters.json").open("w", encoding="utf-8") as file:
                    json.dump(serialized_parameters, file, indent=2)
                    file.write("\n")

            metadata = {
                "experiment_name": experiment_name,
                "profile_name": profile_name,
                "timestamp": timestamp.isoformat(),
                "sweep": _array_metadata(sweep_arrays),
                "results": _array_metadata(result_arrays),
                "parameters": "parameters.json" if serialized_parameters is not None else None,
            }
            with (temporary_directory / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)
                file.write("\n")

            temporary_directory.replace(run_directory)
        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

        return run_directory

    def save_xarray(
        self,
        experiment_name: str,
        dataset: Any,
        profile_name: str | Profile | None = None,
        parameters: Any | None = None,
        now: datetime | None = None,
    ) -> Path:
        """Save all xarray coordinates as sweeps and data variables as results."""
        sweep = {name: coordinate.values for name, coordinate in dataset.coords.items()}
        results = {name: variable.values for name, variable in dataset.data_vars.items()}
        return self.save(
            experiment_name,
            sweep,
            results,
            profile_name=profile_name,
            parameters=parameters,
            now=now,
        )

    def save_figures(self, run_directory: Path | str, figures: Mapping[str, Any]) -> Path:
        """Save named Matplotlib figures into an existing calibration run."""
        run_directory = Path(run_directory)
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Calibration run directory does not exist: {run_directory}")

        timestamp = None
        metadata_path = run_directory / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                timestamp = json.load(file).get("timestamp")

        figures_directory = run_directory / "figures"
        figures_directory.mkdir(exist_ok=True)
        for name, figure in figures.items():
            filename = f"{_validate_name(str(name), 'figure name')}.png"
            add_calibration_timestamp(figure, timestamp)
            figure.savefig(figures_directory / filename, dpi=180, bbox_inches="tight")
        return figures_directory

    def save_analysis_result(self, run_directory: Path | str, analysis_result: Any) -> Path:
        """Save a JSON-serializable analysis result into an existing run."""
        run_directory = Path(run_directory)
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Calibration run directory does not exist: {run_directory}")

        if hasattr(analysis_result, "to_dict"):
            serialized = analysis_result.to_dict()
        else:
            serialized = _to_jsonable(analysis_result)

        output_path = run_directory / "analysis_result.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(_to_jsonable(serialized), file, indent=2, allow_nan=True)
            file.write("\n")

        metadata_path = run_directory / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            metadata["analysis_result"] = output_path.name
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)
                file.write("\n")

        return output_path
