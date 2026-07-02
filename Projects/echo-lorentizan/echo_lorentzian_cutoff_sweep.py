"""Sweep root-Lorentzian cutoff and summarize fitted linewidth and signal."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import replace
import contextlib
import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

QM_LOGGER_NAMES = ("qm", "qm.grpc", "qm.jobs", "qm.api")
for _logger_name in QM_LOGGER_NAMES:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibrations_v2.base import CalibrationOptions
from echo_lorentzian_v2 import EchoLorentzian
from lorentzian import _t2_seconds, plot_raw_data
from parameters import Parameters
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz

DEFAULT_CUTOFFS = np.geomspace(2e-4, 0.99, 20)


def cutoff_points(num_points: int = 10) -> np.ndarray:
    """Return log-spaced cutoff values from 1e-4 through 0.99."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return np.geomspace(2e-3, 0.99, num_points)


def run_cutoff_sweep(
    base_parameters: Parameters | None = None,
    *,
    machine: Quam | None = None,
    qubit: str | None = None,
    cutoffs: Iterable[float] = DEFAULT_CUTOFFS,
    output_root: Path = REPOSITORY_ROOT / "data" / "echo_lorentzian_cutoff_sweep",
    options: CalibrationOptions | None = None,
    auto_connect: bool = True,
) -> dict[str, Any]:
    """Run one echo-Lorentzian spectroscopy experiment per cutoff value."""
    base_parameters = base_parameters or Parameters()
    base_parameters.pulse_shape = "root_lorentzian"
    cutoffs = list(cutoffs)
    output_dir = _new_output_dir(output_root)
    full_records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    output_paths: dict[str, Path | None] = {}
    interrupted = False

    total_cutoffs = len(cutoffs)
    _show_outer_progress(0, total_cutoffs, "starting")
    try:
        for run_index, cutoff in enumerate(cutoffs, start=1):
            parameters = copy.deepcopy(base_parameters)
            parameters.pulse_shape = "root_lorentzian"
            parameters.cutoff = float(cutoff)
            _show_outer_progress(
                run_index - 1,
                total_cutoffs,
                f"running cutoff={float(cutoff):.4g}",
            )
            with _quiet_inner_run():
                calibration = EchoLorentzian(
                    parameters=parameters,
                    options=_individual_run_options(options),
                    machine=machine,
                    qubit=qubit,
                    auto_connect=auto_connect,
                    name=f"echo_lorentzian_cutoff_{run_index:02d}",
                    logger=_quiet_log,
                )
                calibration.run()
            ds = calibration.results["ds_raw"]
            run_records = summarize_cutoff_dataset(
                ds,
                float(cutoff),
                run_index,
                calibration.namespace.get("qubits"),
            )
            full_records.extend(run_records)
            best_records.extend(best_signal_records(run_records))
            run_summaries.append(
                {
                    "run_index": run_index,
                    "cutoff": float(cutoff),
                    "calibration_name": calibration.name,
                }
            )
            _save_individual_figures(
                calibration, ds, output_dir, run_index, float(cutoff)
            )
            _show_outer_progress(
                run_index,
                total_cutoffs,
                f"finished cutoff={float(cutoff):.4g}",
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nCutoff sweep interrupted; saving completed results.")
    print()

    output_paths = _save_sweep_outputs(
        output_dir,
        base_parameters,
        run_summaries,
        full_records,
        best_records,
        interrupted=interrupted,
    )
    return {
        "output_dir": output_dir,
        "fit_results": full_records,
        "best_signal": best_records,
        "figure": output_paths["summary"],
        "fwhm_heatmap": output_paths["fwhm_heatmap"],
        "per_cutoff_traces": output_paths["per_cutoff_traces"],
        "interrupted": interrupted,
    }


def _quiet_log(_message: str) -> None:
    """Drop nested calibration lifecycle logs during the outer sweep."""


@contextlib.contextmanager
def _quiet_inner_run():
    """Suppress SDK logs and printed inner progress while preserving failures."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    saved_levels = {name: logging.getLogger(name).level for name in QM_LOGGER_NAMES}
    saved_disabled = {
        name: logging.getLogger(name).disabled for name in QM_LOGGER_NAMES
    }
    try:
        for name in QM_LOGGER_NAMES:
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)
            logger.disabled = False
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield
    except Exception:
        captured = "\n".join(
            part.strip()
            for part in (stdout.getvalue(), stderr.getvalue())
            if part.strip()
        )
        if captured:
            print(f"\nSuppressed inner-run output before failure:\n{captured[-4000:]}")
        raise
    finally:
        logging.disable(previous_disable_level)
        for name in QM_LOGGER_NAMES:
            logger = logging.getLogger(name)
            logger.setLevel(saved_levels[name])
            logger.disabled = saved_disabled[name]


def _show_outer_progress(completed: int, total: int, label: str) -> None:
    width = 28
    fraction = completed / total if total else 1
    filled = int(round(width * fraction))
    bar = "#" * filled + "." * (width - filled)
    percent = 100 * fraction
    print(
        f"\rCutoff sweep [{bar}] {percent:5.1f}% ({completed}/{total}) {label}",
        end="",
        flush=True,
    )


def _save_individual_figures(
    calibration: EchoLorentzian,
    ds: xr.Dataset,
    output_dir: Path,
    run_index: int,
    cutoff: float,
) -> None:
    """Save per-cutoff figures only, without saving full inner experiment data."""
    qubits = calibration.namespace.get("qubits")
    if not qubits:
        return
    figures = plot_per_qubit(
        plot_raw_data,
        ds,
        qubits,
        figure_name=f"cutoff_{run_index:02d}_{_cutoff_stem(cutoff)}",
        use_state_discrimination=calibration.parameters.use_state_discrimination,
    )
    figures_dir = output_dir / "individual_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for figure_name, figure in figures.items():
        figure.savefig(figures_dir / f"{figure_name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def _cutoff_stem(cutoff: float) -> str:
    return f"cutoff_{cutoff:.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def _save_sweep_outputs(
    output_dir: Path,
    base_parameters: Parameters,
    run_summaries: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
    best_records: list[dict[str, Any]],
    *,
    interrupted: bool,
) -> dict[str, Path | None]:
    _write_csv(output_dir / "cutoff_sweep_fit_results.csv", full_records)
    _write_csv(output_dir / "cutoff_sweep_best_signal.csv", best_records)
    _write_manifest(
        output_dir,
        base_parameters,
        run_summaries,
        interrupted=interrupted,
    )
    return {
        "summary": plot_cutoff_summary(best_records, output_dir),
        "fwhm_heatmap": plot_fwhm_heatmap(full_records, output_dir),
        "per_cutoff_traces": plot_per_cutoff_traces(full_records, output_dir),
    }


def summarize_cutoff_dataset(
    ds: xr.Dataset,
    cutoff: float,
    run_index: int,
    qubits: Any | None = None,
) -> list[dict[str, Any]]:
    """Flatten per-amplitude FWHM and fit signal metrics from one dataset."""
    required = {
        "gaussian_fwhm_hz",
        "gaussian_fit_amplitude",
        "gaussian_fit_abs_amplitude",
        "gaussian_fit_r_squared",
    }
    missing = required - set(ds.variables)
    if missing:
        raise RuntimeError(f"Dataset is missing fitted metrics: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    qubits_by_name = _qubits_by_name(qubits)
    for qubit in ds.qubit.values:
        selected = ds.sel(qubit=qubit)
        qubit_object = qubits_by_name.get(str(qubit))
        t2_s = _record_t2_seconds(qubit_object)
        t2_limit_hz = _t2_fwhm_limit_hz(t2_s)
        for amp_prefactor in ds.amp_prefactor.values:
            point = selected.sel(amp_prefactor=amp_prefactor)
            fwhm_hz = _data_value(point, "gaussian_fwhm_hz")
            full_amp_v = _coord_value(point, "full_amp")
            record = {
                "run_index": run_index,
                "cutoff": cutoff,
                "qubit": str(qubit),
                "amp_prefactor": _finite_float(amp_prefactor),
                "full_amp_v": full_amp_v,
                "rabi_frequency_mhz": _rabi_frequency_mhz(
                    full_amp_v,
                    qubit_object,
                ),
                "gaussian_center_hz": _data_value(point, "gaussian_center_hz"),
                "fwhm_hz": fwhm_hz,
                "fwhm_mhz": fwhm_hz * 1e-6,
                "t2_s": t2_s,
                "t2_fwhm_limit_hz": t2_limit_hz,
                "fwhm_t2_units": _normalized_fwhm(fwhm_hz, t2_limit_hz),
                "fit_amplitude": _data_value(point, "gaussian_fit_amplitude"),
                "fit_abs_amplitude": _data_value(
                    point,
                    "gaussian_fit_abs_amplitude",
                ),
                "fit_r_squared": _data_value(point, "gaussian_fit_r_squared"),
            }
            records.append(record)
    return records


def best_signal_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the valid amplitude with largest fitted signal for each qubit."""
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        fwhm = record["fwhm_hz"]
        signal = record["fit_abs_amplitude"]
        if not _is_finite_number(fwhm) or not _is_finite_number(signal):
            continue
        qubit = record["qubit"]
        previous = best.get(qubit)
        if previous is None or signal > previous["fit_abs_amplitude"]:
            best[qubit] = dict(record)
    return list(best.values())


def _qubits_by_name(qubits: Any | None) -> dict[str, Any]:
    if qubits is None:
        return {}
    return {
        str(getattr(qubit, "name", index)): qubit for index, qubit in enumerate(qubits)
    }


def _record_t2_seconds(qubit: Any | None) -> float:
    if qubit is None:
        return np.nan
    value = _t2_seconds(qubit)
    return float(value) if value is not None and np.isfinite(value) else np.nan


def _rabi_frequency_mhz(full_amp_v: float, qubit: Any | None) -> float:
    if qubit is None or not _is_finite_number(full_amp_v):
        return np.nan
    try:
        pi_pulse = qubit.xy.operations["x180"]
        rabi_hz = amplitude_to_rabi_frequency_hz(
            float(full_amp_v),
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        )
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return np.nan
    return float(rabi_hz) * 1e-6 if np.isfinite(rabi_hz) else np.nan


def _t2_fwhm_limit_hz(t2_s: float) -> float:
    if not _is_finite_number(t2_s) or float(t2_s) <= 0:
        return np.nan
    return 1 / (np.pi * float(t2_s))


def _normalized_fwhm(fwhm_hz: float, t2_limit_hz: float) -> float:
    if (
        not _is_finite_number(fwhm_hz)
        or not _is_finite_number(t2_limit_hz)
        or float(t2_limit_hz) <= 0
    ):
        return np.nan
    return float(fwhm_hz) / float(t2_limit_hz)


def _individual_run_options(options: CalibrationOptions | None) -> CalibrationOptions:
    """Keep nested cutoff experiments in memory; save only sweep-level outputs."""
    base_options = options or CalibrationOptions()
    return replace(
        base_options,
        save_raw_data=False,
        save_analysis_result=False,
        save_figures=False,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )


def plot_cutoff_summary(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Plot FWHM and fitted signal versus cutoff for the best point per qubit."""
    if not records:
        return None

    qubits = sorted({record["qubit"] for record in records})
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for qubit in qubits:
        qubit_records = sorted(
            (record for record in records if record["qubit"] == qubit),
            key=lambda record: record["cutoff"],
        )
        cutoffs = [record["cutoff"] for record in qubit_records]
        axes[0].plot(
            cutoffs,
            [record["fwhm_t2_units"] for record in qubit_records],
            marker="o",
            label=qubit,
        )
        axes[1].plot(
            cutoffs,
            [record["fit_abs_amplitude"] for record in qubit_records],
            marker="o",
            label=qubit,
        )

    axes[0].set_ylabel("FWHM / (1/(pi*T2))")
    axes[1].set_ylabel("Fit signal amplitude")
    axes[1].set_xlabel("Cutoff")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "cutoff_sweep_summary.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def plot_fwhm_heatmap(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Plot FWHM and FWHM/signal as functions of amplitude and cutoff."""
    valid_records = [
        record
        for record in records
        if _is_finite_number(record["cutoff"])
        and _is_finite_number(record["rabi_frequency_mhz"])
        and _is_finite_number(record["fwhm_t2_units"])
    ]
    if not valid_records:
        return None

    qubits = sorted({record["qubit"] for record in valid_records})
    figure, axes = plt.subplots(
        2 * len(qubits),
        1,
        figsize=(8, 7 * len(qubits)),
        squeeze=False,
    )
    for qubit_index, qubit in enumerate(qubits):
        fwhm_ax = axes[2 * qubit_index, 0]
        ratio_ax = axes[2 * qubit_index + 1, 0]
        qubit_records = [record for record in valid_records if record["qubit"] == qubit]
        cutoffs = np.array(sorted({record["cutoff"] for record in qubit_records}))
        rabi_mhz = np.array(
            sorted({record["rabi_frequency_mhz"] for record in qubit_records})
        )
        fwhm = np.full((len(rabi_mhz), len(cutoffs)), np.nan, dtype=float)
        fwhm_over_signal = np.full((len(rabi_mhz), len(cutoffs)), np.nan, dtype=float)
        cutoff_index = {value: index for index, value in enumerate(cutoffs)}
        rabi_index = {value: index for index, value in enumerate(rabi_mhz)}
        for record in qubit_records:
            row = rabi_index[record["rabi_frequency_mhz"]]
            column = cutoff_index[record["cutoff"]]
            fwhm[row, column] = record["fwhm_t2_units"]
            signal = record.get("fit_abs_amplitude", np.nan)
            if _is_finite_number(signal) and float(signal) > 0:
                fwhm_over_signal[row, column] = record["fwhm_t2_units"] / float(signal)

        image = fwhm_ax.pcolormesh(
            _cell_edges_log(cutoffs),
            _cell_edges_linear(rabi_mhz),
            fwhm,
            shading="auto",
        )
        colorbar = figure.colorbar(image, ax=fwhm_ax)
        colorbar.set_label("FWHM / (1/(pi*T2))")
        fwhm_ax.set_xscale("log")
        fwhm_ax.set_xlabel("Cutoff")
        fwhm_ax.set_ylabel("Rabi frequency [MHz]")
        fwhm_ax.set_title(f"{qubit}: FWHM in T2-limit units")

        ratio_image = ratio_ax.pcolormesh(
            _cell_edges_log(cutoffs),
            _cell_edges_linear(rabi_mhz),
            fwhm_over_signal,
            shading="auto",
        )
        ratio_colorbar = figure.colorbar(ratio_image, ax=ratio_ax)
        ratio_colorbar.set_label("FWHM / (signal * 1/(pi*T2))")
        ratio_ax.set_xscale("log")
        ratio_ax.set_xlabel("Cutoff")
        ratio_ax.set_ylabel("Rabi frequency [MHz]")
        ratio_ax.set_title(f"{qubit}: normalized FWHM divided by fit signal")

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "cutoff_sweep_fwhm_heatmap.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def plot_per_cutoff_traces(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    """Plot the fitted FWHM and signal traces calculated for every cutoff."""
    valid_records = [
        record
        for record in records
        if _is_finite_number(record["cutoff"])
        and _is_finite_number(record["rabi_frequency_mhz"])
    ]
    if not valid_records:
        return None

    qubits = sorted({record["qubit"] for record in valid_records})
    figure, axes = plt.subplots(
        2 * len(qubits),
        1,
        figsize=(8, 7 * len(qubits)),
        squeeze=False,
    )
    for qubit_index, qubit in enumerate(qubits):
        fwhm_ax = axes[2 * qubit_index, 0]
        signal_ax = axes[2 * qubit_index + 1, 0]
        qubit_records = [record for record in valid_records if record["qubit"] == qubit]
        cutoffs = np.array(sorted({record["cutoff"] for record in qubit_records}))
        colors = plt.cm.viridis(np.linspace(0, 1, len(cutoffs)))

        for color, cutoff in zip(colors, cutoffs):
            cutoff_records = sorted(
                (record for record in qubit_records if record["cutoff"] == cutoff),
                key=lambda record: record["rabi_frequency_mhz"],
            )
            x = [record["rabi_frequency_mhz"] for record in cutoff_records]
            fwhm_ax.plot(
                x,
                [record["fwhm_t2_units"] for record in cutoff_records],
                marker="o",
                linewidth=1.2,
                markersize=3,
                color=color,
                label=f"{cutoff:.3g}",
            )
            signal_ax.plot(
                x,
                [record["fit_abs_amplitude"] for record in cutoff_records],
                marker="o",
                linewidth=1.2,
                markersize=3,
                color=color,
                label=f"{cutoff:.3g}",
            )

        fwhm_ax.set_title(f"{qubit}: fitted FWHM trace for each cutoff")
        fwhm_ax.set_ylabel("FWHM / (1/(pi*T2))")
        signal_ax.set_title(f"{qubit}: fitted signal trace for each cutoff")
        signal_ax.set_ylabel("Fit signal amplitude")
        signal_ax.set_xlabel("Rabi frequency [MHz]")
        for axis in (fwhm_ax, signal_ax):
            axis.grid(alpha=0.25)
            axis.legend(title="cutoff", fontsize=7, ncols=2)

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "cutoff_sweep_per_cutoff_traces.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def _cell_edges_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] / np.sqrt(10), values[0] * np.sqrt(10)])
    log_values = np.log10(values)
    log_edges = _cell_edges_linear(log_values)
    return 10**log_edges


def _cell_edges_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = 0.5 if values[0] == 0 else abs(values[0]) * 0.5
        return np.array([values[0] - delta, values[0] + delta])
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def _new_output_dir(output_root: Path) -> Path:
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_manifest(
    output_dir: Path,
    base_parameters: Parameters,
    runs: list[dict[str, Any]],
    *,
    interrupted: bool = False,
) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "interrupted": bool(interrupted),
        "completed_runs": len(runs),
        "cutoffs": [run["cutoff"] for run in runs],
        "runs": runs,
        "base_parameters": {
            key: _jsonable(value)
            for key, value in vars(base_parameters).items()
            if not key.startswith("_")
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _coord_value(point: xr.Dataset, name: str) -> float:
    if name not in point.coords:
        return np.nan
    return _finite_float(point.coords[name].values)


def _data_value(point: xr.Dataset, name: str, scale: float = 1.0) -> float:
    if name not in point:
        return np.nan
    return _finite_float(point[name].values) * scale


def _finite_float(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        array = array.reshape(-1)[:1]
    scalar = float(array.item())
    return scalar if np.isfinite(scalar) else np.nan


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = True
    parameters.num_shots = 60
    parameters.lorentzian_length_in_ns = 20000
    parameters.waveform_template_length_in_ns = 20000
    parameters.lorentzian_peak_amplitude = 0.2
    parameters.min_amp_factor = 0.0
    parameters.max_amp_factor = 1
    parameters.amp_factor_step = 0.04
    parameters.frequency_span_in_mhz = 5
    parameters.frequency_step_in_mhz = 0.005

    result = run_cutoff_sweep(
        parameters,
        machine=create_machine(qubit="q1"),
        qubit="q1",
        cutoffs=cutoff_points(20),
    )
    print(f"Cutoff sweep results saved to {result['output_dir']}")
