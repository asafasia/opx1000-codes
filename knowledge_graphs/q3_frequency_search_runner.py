from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from quam_config import create_machine
from calibrations_v2.base import CalibrationOptions


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_DATA_ROOT = ROOT / "knowledge_graphs" / "automation_data"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resonator_module = _load_module(
    ROOT / "calibrations_v2" / "02a_resonator_spectroscopy.py",
    "q3_resonator_spectroscopy_module",
)
qubit_module = _load_module(
    ROOT / "calibrations_v2" / "03a_qubit_spectroscopy.py",
    "q3_qubit_spectroscopy_module",
)
power_rabi_module = _load_module(
    ROOT / "calibrations_v2" / "04b_power_rabi.py",
    "q3_power_rabi_module",
)
power_rabi_chevron_module = _load_module(
    ROOT / "calibrations_v2" / "04d_power_rabi_chevron.py",
    "q3_power_rabi_chevron_module",
)


def _options() -> CalibrationOptions:
    return CalibrationOptions(
        save_raw_data=True,
        save_analysis_result=True,
        save_figures=True,
        analyse_data=True,
        plot_data=True,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )


def _set_qubit_frequency(machine: Any, frequency_hz: float) -> None:
    selected_qubits = list(machine.qubits.keys())
    if len(selected_qubits) != 1:
        raise ValueError(f"Expected one selected qubit, got {selected_qubits}")
    qubit = machine.qubits[selected_qubits[0]]
    old_if = float(qubit.xy.intermediate_frequency)
    delta = float(frequency_hz) - float(qubit.xy.RF_frequency)
    qubit.f_01 = float(frequency_hz)
    qubit.xy.RF_frequency = float(frequency_hz)
    if isinstance(qubit.xy.get_reference("intermediate_frequency", None), str):
        qubit.xy.intermediate_frequency = None
    qubit.xy.intermediate_frequency = old_if + delta


def _metric_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _selected_qubit_name(calibration: Any) -> str:
    qubits = list(calibration.results["ds_raw"].qubit.values)
    if len(qubits) != 1:
        raise ValueError(f"Expected one qubit in dataset, got {qubits}")
    return str(qubits[0])


def _resonator_summary(calibration: Any) -> dict[str, Any]:
    qubit_name = _selected_qubit_name(calibration)
    ds = calibration.results["ds_raw"].sel(qubit=qubit_name)
    sep = ds.IQ_separation
    best_index = int(sep.argmax(dim="detuning").values)
    best_sep = _metric_float(sep.isel(detuning=best_index).values)
    median_sep = _metric_float(sep.median(dim="detuning").values)
    max_minus_median = (
        None if best_sep is None or median_sep is None else best_sep - median_sep
    )
    return {
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "best_separation": best_sep,
        "median_separation": median_sep,
        "best_minus_median_separation": max_minus_median,
        "best_detuning_hz": _metric_float(sep.detuning.isel(detuning=best_index).values),
        "fit_results": calibration.results.get("fit_results", {}).get(qubit_name, {}),
        "outcome": calibration.outcomes.get(qubit_name),
    }


def _qubit_summary(calibration: Any) -> dict[str, Any]:
    qubit_name = _selected_qubit_name(calibration)
    fit_results = calibration.results.get("fit_results", {}).get(qubit_name, {})
    ds_fit = calibration.results.get("ds_fit")
    summary = {
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "fit_results": fit_results,
        "outcome": calibration.outcomes.get(qubit_name),
    }
    if ds_fit is not None:
        qfit = ds_fit.sel(qubit=qubit_name)
        for name in [
            "fit_r_squared",
            "fit_amplitude",
            "relative_freq",
            "fwhm",
            "measured_max_position",
        ]:
            if name in qfit:
                summary[name] = _metric_float(qfit[name].values)
    return summary


def _power_rabi_summary(calibration: Any) -> dict[str, Any]:
    qubit_name = _selected_qubit_name(calibration)
    fit_results = calibration.results.get("fit_results", {}).get(qubit_name, {})
    return {
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "fit_results": fit_results,
        "outcome": calibration.outcomes.get(qubit_name),
    }


def _power_rabi_chevron_summary(calibration: Any) -> dict[str, Any]:
    qubit_name = _selected_qubit_name(calibration)
    ds = calibration.results.get("ds_raw")
    summary = {
        "run_directory": str(calibration.namespace.get("calibration_run_directory", "")),
        "outcome": calibration.outcomes.get(qubit_name),
    }
    if ds is not None:
        qds = ds.sel(qubit=qubit_name)
        summary["num_detuning_points"] = int(qds.sizes.get("detuning", 0))
        summary["num_amp_points"] = int(qds.sizes.get("amp_prefactor", 0))
        if "full_freq" in qds.coords:
            freqs = qds.full_freq.values
            summary["min_frequency_hz"] = _metric_float(np.nanmin(freqs))
            summary["max_frequency_hz"] = _metric_float(np.nanmax(freqs))
        if "amp_prefactor" in qds.coords:
            amps = qds.amp_prefactor.values
            summary["min_amp_factor"] = _metric_float(np.nanmin(amps))
            summary["max_amp_factor"] = _metric_float(np.nanmax(amps))
    return summary


def _run_artifact_name(record: dict[str, Any]) -> str:
    run_directory = Path(record.get("summary", {}).get("run_directory", ""))
    run_id = run_directory.name if run_directory.name else "unknown_run"
    center_hz = int(float(record["center_frequency_hz"]))
    return f"{run_id}_{record['kind']}_{center_hz}_amp_{record['amplitude_factor']}"


def _copy_figures(source_directory: Path, target_directory: Path) -> list[str]:
    try:
        if not source_directory.exists():
            return []
    except OSError:
        return []
    target_directory.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    try:
        sources = sorted(source_directory.iterdir())
    except OSError:
        return []
    for source in sources:
        if not source.is_file():
            continue
        target = target_directory / source.name
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        copied.append(str(target.relative_to(ROOT)))
    return copied


def _publish_record_artifacts(qubit_name: str, record: dict[str, Any]) -> None:
    artifact_dir = (
        AUTOMATION_DATA_ROOT
        / qubit_name
        / "frequency_discovery"
        / _run_artifact_name(record)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_directory_value = record.get("summary", {}).get("run_directory", "")
    figure_files = []
    if run_directory_value:
        run_directory = Path(run_directory_value)
        figure_files = _copy_figures(run_directory / "figures", artifact_dir / "figures")

    record["artifact_directory"] = str(artifact_dir.relative_to(ROOT))
    record["figure_directory"] = str((artifact_dir / "figures").relative_to(ROOT))
    record["figure_files"] = figure_files

    with (artifact_dir / "record.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_record(qubit_name: str, record: dict[str, Any]) -> None:
    _publish_record_artifacts(qubit_name, record)
    log_path = ROOT / "knowledge_graphs" / f"{qubit_name}_frequency_search_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_resonator(qubit_name: str, center_frequency_hz: float, amplitude_factor: float, shots: int) -> dict[str, Any]:
    machine = create_machine(qubit=qubit_name)
    _set_qubit_frequency(machine, center_frequency_hz)
    params = resonator_module.Parameters()
    params.qubit_operation = "saturation"
    params.saturation_amplitude_factor = amplitude_factor
    params.saturation_lead_time_in_ns = 6_000
    params.num_shots = shots
    params.frequency_span_in_mhz = 30
    params.frequency_step_in_mhz = params.frequency_span_in_mhz / 200
    params.timeout = 120
    calibration = resonator_module.ResonatorSpectroscopy(
        parameters=params,
        options=_options(),
        machine=machine,
    )
    status = calibration.run()
    summary = _resonator_summary(calibration)
    record = {
        "kind": "resonator",
        "qubit": qubit_name,
        "center_frequency_hz": center_frequency_hz,
        "amplitude_factor": amplitude_factor,
        "shots": shots,
        "status": status.__dict__,
        "summary": summary,
    }
    _append_record(qubit_name, record)
    return record


def run_qubit(
    qubit_name: str,
    center_frequency_hz: float,
    amplitude_factor: float,
    span_mhz: float,
    shots: int,
) -> dict[str, Any]:
    machine = create_machine(qubit=qubit_name)
    _set_qubit_frequency(machine, center_frequency_hz)
    params = qubit_module.Parameters()
    params.use_state_discrimination = False
    params.reset_type = "thermal"
    params.operation = "saturation"
    params.operation_amplitude_factor = amplitude_factor
    params.frequency_span_in_mhz = span_mhz
    params.frequency_step_in_mhz = params.frequency_span_in_mhz / 200
    params.num_shots = shots
    params.timeout = 180
    calibration = qubit_module.QubitSpectroscopy(
        parameters=params,
        options=_options(),
        machine=machine,
    )
    status = calibration.run()
    summary = _qubit_summary(calibration)
    record = {
        "kind": "qubit",
        "qubit": qubit_name,
        "center_frequency_hz": center_frequency_hz,
        "amplitude_factor": amplitude_factor,
        "span_mhz": span_mhz,
        "step_mhz": params.frequency_step_in_mhz,
        "shots": shots,
        "status": status.__dict__,
        "summary": summary,
    }
    _append_record(qubit_name, record)
    return record


def run_power_rabi(
    qubit_name: str,
    center_frequency_hz: float,
    max_amp_factor: float,
    amp_step: float,
    shots: int,
    pi_repetitions: int = 3,
) -> dict[str, Any]:
    machine = create_machine(qubit=qubit_name)
    _set_qubit_frequency(machine, center_frequency_hz)
    params = power_rabi_module.Parameters()
    params.reset_type = "thermal"
    params.use_state_discrimination = False
    params.transition = "ge"
    params.operation = "x180"
    params.pi_repetitions = pi_repetitions
    params.min_amp_factor = 0.0
    params.max_amp_factor = max_amp_factor
    params.amp_factor_step = amp_step
    params.num_shots = shots
    params.timeout = 180
    calibration = power_rabi_module.PowerRabi(
        parameters=params,
        options=_options(),
        machine=machine,
    )
    status = calibration.run()
    summary = _power_rabi_summary(calibration)
    record = {
        "kind": "power_rabi",
        "qubit": qubit_name,
        "center_frequency_hz": center_frequency_hz,
        "amplitude_factor": max_amp_factor,
        "amp_step": amp_step,
        "pi_repetitions": pi_repetitions,
        "shots": shots,
        "status": status.__dict__,
        "summary": summary,
    }
    _append_record(qubit_name, record)
    return record


def run_power_rabi_chevron(
    qubit_name: str,
    center_frequency_hz: float,
    max_amp_factor: float,
    amp_step: float,
    span_mhz: float,
    shots: int,
    operation: str = "x180",
) -> dict[str, Any]:
    machine = create_machine(qubit=qubit_name)
    _set_qubit_frequency(machine, center_frequency_hz)
    params = power_rabi_chevron_module.Parameters()
    params.reset_type = "thermal"
    params.use_state_discrimination = False
    params.operation = operation
    params.min_amp_factor = 0.0
    params.max_amp_factor = max_amp_factor
    params.amp_factor_step = amp_step
    params.frequency_span_in_mhz = span_mhz
    params.frequency_step_in_mhz = span_mhz / 80
    params.num_shots = shots
    params.timeout = 240
    calibration = power_rabi_chevron_module.PowerRabiChevron(
        parameters=params,
        options=_options(),
        machine=machine,
    )
    status = calibration.run()
    summary = _power_rabi_chevron_summary(calibration)
    record = {
        "kind": "power_rabi_chevron",
        "qubit": qubit_name,
        "center_frequency_hz": center_frequency_hz,
        "amplitude_factor": max_amp_factor,
        "amp_step": amp_step,
        "span_mhz": span_mhz,
        "step_mhz": params.frequency_step_in_mhz,
        "operation": operation,
        "shots": shots,
        "status": status.__dict__,
        "summary": summary,
    }
    _append_record(qubit_name, record)
    return record


def _listed_frequency_hz(qubit_name: str) -> float:
    machine = create_machine(qubit=qubit_name)
    return float(machine.qubits[qubit_name].f_01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=["resonator", "qubit", "rabi", "chevron"],
        help="Calibration type to execute.",
    )
    parser.add_argument("--qubit", default="q3")
    parser.add_argument("--center-hz", type=float)
    parser.add_argument("--amp", type=float, default=0.7)
    parser.add_argument("--amp-step", type=float, default=0.01)
    parser.add_argument("--pi-repetitions", type=int, default=3)
    parser.add_argument("--span-mhz", type=float, default=300.0)
    parser.add_argument("--shots", type=int, default=150)
    args = parser.parse_args()
    center_hz = args.center_hz if args.center_hz is not None else _listed_frequency_hz(args.qubit)

    if args.kind == "resonator":
        record = run_resonator(args.qubit, center_hz, args.amp, args.shots)
    elif args.kind == "qubit":
        record = run_qubit(
            args.qubit,
            center_hz,
            args.amp,
            args.span_mhz,
            args.shots,
        )
    elif args.kind == "rabi":
        record = run_power_rabi(
            args.qubit,
            center_hz,
            args.amp,
            args.amp_step,
            args.shots,
            args.pi_repetitions,
        )
    else:
        record = run_power_rabi_chevron(
            args.qubit,
            center_hz,
            args.amp,
            args.amp_step,
            args.span_mhz,
            args.shots,
        )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
