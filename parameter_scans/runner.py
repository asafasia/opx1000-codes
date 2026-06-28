"""Run calibration scripts repeatedly and save only fitted parameters.

The runner is designed for unattended overnight/weekend drift monitoring. It
executes existing calibration scripts, extracts compact fit summaries from the
global ``node`` object each script creates, and writes results incrementally
under ``data/parameter_scans``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "parameter_scans"
DEFAULT_STOP_FILE = "STOP"
LEGACY_CALIBRATIONS_DIR = "calibrations"
V2_CALIBRATIONS_DIR = "calibrations_v2"
SUMMARY_FIELDS = [
    "timestamp",
    "cycle",
    "experiment_name",
    "script",
    "status",
    "qubit",
    "parameter",
    "value",
    "unit",
    "success",
    "duration_s",
    "error",
]

PARAMETER_ALIASES = {
    "t1": ("T1", "ns"),
    "t1_error": ("T1 error", "ns"),
    "decay": ("decay", ""),
    "decay_error": ("decay error", ""),
    "freq_offset": ("frequency offset", "Hz"),
    "frequency": ("qubit frequency", "Hz"),
    "relative_freq": ("relative frequency", "Hz"),
    "fwhm": ("FWHM", "Hz"),
    "iw_angle": ("integration weight angle", "rad"),
    "saturation_amp": ("saturation amplitude", "V"),
    "x180_amp": ("x180 amplitude", "V"),
}


@dataclass
class ExperimentSpec:
    """One calibration script in a long scan plan."""

    script: Path
    name: str | None = None
    enabled: bool = True

    @classmethod
    def from_config(cls, value: str | Mapping[str, Any]) -> "ExperimentSpec":
        if isinstance(value, str):
            return cls(script=Path(value))
        return cls(
            script=Path(str(value["script"])),
            name=str(value["name"]) if value.get("name") else None,
            enabled=bool(value.get("enabled", True)),
        )


@dataclass
class ScanConfig:
    """Configuration for an unattended parameter scan."""

    name: str = "parameter_scan"
    experiments: list[ExperimentSpec] = field(default_factory=list)
    repetitions: int | None = 1
    interval_seconds: float = 0.0
    continue_on_error: bool = False
    save_full_results: bool = False
    profile_name: str | None = None
    qubit: str | None = None
    stop_file: str = DEFAULT_STOP_FILE
    output_root: Path = DEFAULT_OUTPUT_ROOT


def load_scan_config(path: Path | str) -> ScanConfig:
    """Load a scan plan from JSON."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ScanConfig(
        name=str(payload.get("name", config_path.stem)),
        experiments=[ExperimentSpec.from_config(item) for item in payload.get("experiments", [])],
        repetitions=payload.get("repetitions", 1),
        interval_seconds=float(payload.get("interval_seconds", 0.0)),
        continue_on_error=bool(payload.get("continue_on_error", False)),
        save_full_results=bool(payload.get("save_full_results", False)),
        profile_name=payload.get("profile_name"),
        qubit=payload.get("qubit"),
        stop_file=str(payload.get("stop_file", DEFAULT_STOP_FILE)),
        output_root=Path(payload.get("output_root", DEFAULT_OUTPUT_ROOT)),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _experiment_name(script: Path, node: Any | None, spec_name: str | None) -> str:
    if spec_name:
        return spec_name
    node_name = getattr(node, "name", None)
    return str(node_name) if node_name else script.stem


def _result_node(namespace: Mapping[str, Any]) -> Any | None:
    """Return the object that exposes node-like calibration results."""
    node = namespace.get("node")
    if node is not None:
        return node
    calibration = namespace.get("calibration")
    if calibration is not None and hasattr(calibration, "results"):
        return calibration
    return None


def _parameter_label(experiment_name: str, key: str) -> tuple[str, str]:
    if key == "decay" and "ramsey" in experiment_name.lower():
        return "T2 Ramsey", "s"
    if key == "decay_error" and "ramsey" in experiment_name.lower():
        return "T2 Ramsey error", "s"
    return PARAMETER_ALIASES.get(key, (key, ""))


def extract_fit_records(
    node: Any,
    *,
    timestamp: str,
    cycle: int,
    experiment_name: str,
    script: Path,
    duration_s: float,
) -> list[dict[str, Any]]:
    """Extract compact per-qubit numeric fit records from a Qualibrate node."""
    fit_results = getattr(node, "results", {}).get("fit_results", {})
    outcomes = getattr(node, "outcomes", {}) or {}
    records: list[dict[str, Any]] = []

    for qubit, raw_parameters in fit_results.items():
        parameters = _plain_mapping(raw_parameters)
        success = parameters.get("success", outcomes.get(qubit))
        for key, raw_value in parameters.items():
            if key == "success":
                continue
            value = _numeric(raw_value)
            if value is None:
                continue
            parameter, unit = _parameter_label(experiment_name, key)
            records.append(
                {
                    "timestamp": timestamp,
                    "cycle": cycle,
                    "experiment_name": experiment_name,
                    "script": script.as_posix(),
                    "status": "ok",
                    "qubit": str(qubit),
                    "parameter": parameter,
                    "value": value,
                    "unit": unit,
                    "success": bool(success) if isinstance(success, (bool, np.bool_)) else str(success),
                    "duration_s": round(duration_s, 3),
                    "error": "",
                }
            )
    return records


class LongScanRunner:
    """Execute a scan plan and persist summaries after every experiment."""

    def __init__(self, config: ScanConfig, repository_root: Path | str = REPOSITORY_ROOT) -> None:
        self.config = config
        self.repository_root = Path(repository_root)
        self.run_directory = self._make_run_directory()
        self.summary_path = self.run_directory / "summary.csv"
        self.events_path = self.run_directory / "events.jsonl"
        self.status_path = self.run_directory / "status.json"
        self.records: list[dict[str, Any]] = []
        self._skipped_save_index = 0

    def _make_run_directory(self) -> Path:
        now = datetime.now().astimezone()
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in self.config.name)
        run_directory = (
            self.config.output_root
            / now.strftime("%Y-%m-%d")
            / safe_name
            / now.strftime("%H-%M-%S-%f")
        )
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    def _write_manifest(self) -> None:
        manifest = {
            "name": self.config.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "repetitions": self.config.repetitions,
            "interval_seconds": self.config.interval_seconds,
            "continue_on_error": self.config.continue_on_error,
            "save_full_results": self.config.save_full_results,
            "profile_name": self.config.profile_name,
            "qubit": self.config.qubit,
            "stop_file": self.config.stop_file,
            "experiments": [
                {
                    "script": spec.script.as_posix(),
                    "name": spec.name,
                    "enabled": spec.enabled,
                }
                for spec in self.config.experiments
            ],
        }
        _atomic_json(self.run_directory / "scan.json", manifest)

    def _append_event(self, event: Mapping[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(event), default=_json_default) + "\n")

    def _write_summary(self) -> None:
        temporary = self.summary_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows({field: record.get(field, "") for field in SUMMARY_FIELDS} for record in self.records)
        temporary.replace(self.summary_path)
        _atomic_json(self.run_directory / "summary.json", self.records)

    def _write_status(self, status: str, message: str = "") -> None:
        payload = {
            "status": status,
            "message": message,
            "updated_at": datetime.now().astimezone().isoformat(),
            "records": len(self.records),
            "run_directory": str(self.run_directory),
        }
        _atomic_json(self.status_path, payload)

    def _stop_requested(self) -> bool:
        return (self.run_directory / self.config.stop_file).exists()

    def _sleep_between_cycles(self) -> bool:
        deadline = time.monotonic() + self.config.interval_seconds
        while time.monotonic() < deadline:
            if self._stop_requested():
                return False
            time.sleep(min(1.0, deadline - time.monotonic()))
        return True

    def _script_path(self, spec: ExperimentSpec) -> Path:
        path = spec.script if spec.script.is_absolute() else self.repository_root / spec.script
        if not path.is_file():
            path = self._v2_script_path(spec.script, path)
        if not path.is_file():
            raise FileNotFoundError(f"Experiment script does not exist: {path}")
        return path

    def _v2_script_path(self, requested_script: Path, resolved_path: Path) -> Path:
        """Map legacy ``calibrations/*.py`` requests to ``calibrations_v2/*.py``.

        Older scan plans and the live app were written against the original
        ``calibrations`` directory. The current calibration language lives in
        ``calibrations_v2``; keeping this resolver here lets those scan plans
        continue to work while newly written configs can point at v2 directly.
        """
        parts = requested_script.parts
        if parts and parts[0] == LEGACY_CALIBRATIONS_DIR:
            candidate = self.repository_root / V2_CALIBRATIONS_DIR / Path(*parts[1:])
            if candidate.is_file():
                return candidate
        if resolved_path.parent.name == LEGACY_CALIBRATIONS_DIR:
            candidate = resolved_path.parent.parent / V2_CALIBRATIONS_DIR / resolved_path.name
            if candidate.is_file():
                return candidate
        return resolved_path

    def _install_summary_only_saver(self) -> Any | None:
        if self.config.save_full_results:
            return None
        try:
            from calibration_io.calibration_saver import CalibrationSaver
        except ImportError:
            return None

        original_save = CalibrationSaver.save
        original_save_xarray = CalibrationSaver.save_xarray
        original_save_figures = CalibrationSaver.save_figures
        skipped_root = self.run_directory / "skipped_raw_calibration_saves"
        skipped_root.mkdir(exist_ok=True)

        def marker(instance: Any, experiment_name: str, *args: Any, **kwargs: Any) -> Path:
            self._skipped_save_index += 1
            directory = skipped_root / f"{self._skipped_save_index:04d}_{experiment_name}"
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                directory / "metadata.json",
                {
                    "experiment_name": experiment_name,
                    "skipped_by": "parameter_scans",
                    "reason": "Long scans store compact fit summaries only.",
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
            )
            return directory

        def save(instance: Any, experiment_name: str, *args: Any, **kwargs: Any) -> Path:
            return marker(instance, str(experiment_name), *args, **kwargs)

        def save_xarray(instance: Any, experiment_name: str, *args: Any, **kwargs: Any) -> Path:
            return marker(instance, str(experiment_name), *args, **kwargs)

        def save_figures(instance: Any, run_directory: Path | str, figures: Mapping[str, Any]) -> Path:
            directory = Path(run_directory) / "figures_skipped"
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                directory / "metadata.json",
                {
                    "skipped_by": "parameter_scans",
                    "figure_count": len(figures),
                    "reason": "Long scans store compact fit summaries only.",
                },
            )
            return directory

        CalibrationSaver.save = save
        CalibrationSaver.save_xarray = save_xarray
        CalibrationSaver.save_figures = save_figures
        return CalibrationSaver, original_save, original_save_xarray, original_save_figures

    def _restore_summary_only_saver(self, patch: Any | None) -> None:
        if patch is None:
            return
        CalibrationSaver, original_save, original_save_xarray, original_save_figures = patch
        CalibrationSaver.save = original_save
        CalibrationSaver.save_xarray = original_save_xarray
        CalibrationSaver.save_figures = original_save_figures

    def _install_v2_scan_options(self) -> Any | None:
        try:
            from calibrations_v2.base import BaseCalibration
        except ImportError:
            return None

        original_init = BaseCalibration.__init__
        original_save_figures = BaseCalibration.save_figures
        original_propose = BaseCalibration._propose_profile_update_from_options

        def init(instance: Any, *args: Any, **kwargs: Any) -> None:
            original_init(instance, *args, **kwargs)
            if not self.config.save_full_results:
                instance.options.save_raw_data = False
                instance.options.save_figures = False
                instance.options.plot_data = False
            instance.options.update_state = False
            instance.options.propose_profile_update = False
            instance.options.apply_profile_update = False

        def save_figures(instance: Any) -> bool:
            if self.config.save_full_results:
                return original_save_figures(instance)
            return False

        def propose(instance: Any) -> bool:
            return False

        BaseCalibration.__init__ = init
        BaseCalibration.save_figures = save_figures
        BaseCalibration._propose_profile_update_from_options = propose
        return BaseCalibration, original_init, original_save_figures, original_propose

    def _restore_v2_scan_options(self, patch: Any | None) -> None:
        if patch is None:
            return
        BaseCalibration, original_init, original_save_figures, original_propose = patch
        BaseCalibration.__init__ = original_init
        BaseCalibration.save_figures = original_save_figures
        BaseCalibration._propose_profile_update_from_options = original_propose

    def _install_plot_guard(self) -> Any | None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        plt.close("all")
        return plt, original_show

    def _restore_plot_guard(self, patch: Any | None) -> None:
        if patch is None:
            return
        plt, original_show = patch
        plt.show = original_show
        plt.close("all")

    def _install_scan_patches(self) -> Callable[[], None]:
        plot_patch = self._install_plot_guard()
        saver_patch = self._install_summary_only_saver()
        v2_patch = self._install_v2_scan_options()

        def restore() -> None:
            self._restore_v2_scan_options(v2_patch)
            self._restore_summary_only_saver(saver_patch)
            self._restore_plot_guard(plot_patch)

        return restore

    def _script_environment(self) -> dict[str, str]:
        environment = {"MPLBACKEND": os.environ.get("MPLBACKEND", "Agg")}
        if self.config.profile_name:
            environment["QUAM_PROFILE"] = self.config.profile_name
        if self.config.qubit:
            environment["QUAM_QUBIT"] = self.config.qubit
        return environment

    def _install_script_environment(self) -> dict[str, str | None]:
        updates = self._script_environment()
        previous = {name: os.environ.get(name) for name in updates}
        os.environ.update(updates)
        return previous

    def _restore_script_environment(self, previous: Mapping[str, str | None]) -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _run_script(self, spec: ExperimentSpec, cycle: int) -> list[dict[str, Any]]:
        script = self._script_path(spec)
        started = time.monotonic()
        timestamp = datetime.now().astimezone().isoformat()

        environment_patch = self._install_script_environment()
        restore_scan_patches = self._install_scan_patches()
        try:
            namespace = runpy.run_path(str(script), run_name="__main__")
        finally:
            restore_scan_patches()
            self._restore_script_environment(environment_patch)
        duration_s = time.monotonic() - started
        node = _result_node(namespace)
        experiment_name = _experiment_name(spec.script, node, spec.name)
        if node is None:
            raise RuntimeError(
                f"{script} did not leave a global 'node' or v2 'calibration' object to inspect"
            )

        records = extract_fit_records(
            node,
            timestamp=timestamp,
            cycle=cycle,
            experiment_name=experiment_name,
            script=spec.script,
            duration_s=duration_s,
        )
        if not records:
            raise RuntimeError(f"{experiment_name} finished but no numeric fit_results were found")
        return records

    def run(self) -> Path:
        """Run the configured scan and return the created run directory."""
        experiments = [spec for spec in self.config.experiments if spec.enabled]
        if not experiments:
            raise ValueError("At least one enabled experiment is required")

        self._write_manifest()
        self._write_summary()
        self._write_status("running")
        max_cycles = self.config.repetitions
        cycle = 0

        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                for spec in experiments:
                    if self._stop_requested():
                        self._write_status("stopped", "Stop file requested")
                        return self.run_directory
                    try:
                        records = self._run_script(spec, cycle)
                        self.records.extend(records)
                        self._append_event(
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "cycle": cycle,
                                "script": spec.script.as_posix(),
                                "status": "ok",
                                "records": len(records),
                            }
                        )
                    except Exception as error:
                        error_text = "".join(traceback.format_exception_only(type(error), error)).strip()
                        failure = {
                            "timestamp": datetime.now().astimezone().isoformat(),
                            "cycle": cycle,
                            "experiment_name": spec.name or spec.script.stem,
                            "script": spec.script.as_posix(),
                            "status": "error",
                            "qubit": "",
                            "parameter": "",
                            "value": "",
                            "unit": "",
                            "success": False,
                            "duration_s": "",
                            "error": error_text,
                        }
                        self.records.append(failure)
                        self._append_event({**failure, "traceback": traceback.format_exc()})
                        self._write_summary()
                        if not self.config.continue_on_error:
                            self._write_status("failed", error_text)
                            return self.run_directory
                    self._write_summary()
                    self._write_status("running")

                if max_cycles is None or cycle < max_cycles:
                    if self.config.interval_seconds > 0:
                        if not self._sleep_between_cycles():
                            self._write_status("stopped", "Stop file requested")
                            return self.run_directory
        except KeyboardInterrupt:
            self._write_status("stopped", "KeyboardInterrupt")
            return self.run_directory

        self._write_status("complete")
        return self.run_directory


def _config_from_args(args: argparse.Namespace) -> ScanConfig:
    if args.config:
        config = load_scan_config(args.config)
    else:
        config = ScanConfig()
    if args.name:
        config.name = args.name
    if args.experiment:
        config.experiments = [ExperimentSpec.from_config(item) for item in args.experiment]
    if args.repetitions is not None:
        config.repetitions = None if args.repetitions == 0 else args.repetitions
    if args.interval_seconds is not None:
        config.interval_seconds = args.interval_seconds
    if args.continue_on_error:
        config.continue_on_error = True
    if args.save_full_results:
        config.save_full_results = True
    if args.profile_name:
        config.profile_name = args.profile_name
    if args.qubit:
        config.qubit = args.qubit
    if args.output_root:
        config.output_root = Path(args.output_root)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibration scripts in a long parameter scan.")
    parser.add_argument("--config", type=Path, help="JSON scan plan. See parameter_scans/example_scan.json.")
    parser.add_argument("--name", help="Scan name used in data/parameter_scans.")
    parser.add_argument("--experiment", action="append", help="Calibration script to run. Can be repeated.")
    parser.add_argument("--repetitions", type=int, help="Number of cycles. Use 0 to run until stopped.")
    parser.add_argument("--interval-seconds", type=float, help="Delay between cycles.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record errors and keep running.")
    parser.add_argument("--save-full-results", action="store_true", help="Allow calibration scripts to save raw data.")
    parser.add_argument("--profile-name", help="Profile to use while running scripts, for example single_qubit.")
    parser.add_argument("--qubit", help="Selected qubit to pass through QUAM_QUBIT.")
    parser.add_argument("--output-root", type=Path, help="Override output root.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    runner = LongScanRunner(config)
    run_directory = runner.run()
    print(f"Parameter scan saved to {run_directory}")
