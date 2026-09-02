"""Compare measured pulse-length-sweep FWHMs with matched qutrit simulations.

Pass the campaign directory written by ``run_pulse_length_sweeps.py``.  The
analysis loads every successful hardware run, extracts its detuning-trace FWHM,
simulates the same detuning and amplitude axes, and applies the identical FWHM
method to the simulated traces. The default robust direct half-height method
rejects clipped or unresolved features; ``--fit-method gaussian`` retains the
slower shared Gaussian model as an optional comparison.

Simulation results are cached per pulse length, so an interrupted analysis can
be resumed without repeating completed lengths.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
for path in (PROJECT_ROOT, REPOSITORY_ROOT, SIMULATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shaped_pulse_spectroscopy.fwhm import add_gaussian_fwhm_analysis
from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--results",
        choices=["auto", "raw", "mitigated"],
        default="auto",
        help="Use mitigated results when present, raw results, or auto-select.",
    )
    parser.add_argument(
        "--simulation-steps-per-us",
        type=int,
        default=800,
        help="RK4 steps per microsecond in each half-pulse (paper setting: 800).",
    )
    parser.add_argument(
        "--minimum-steps-per-half",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--force-simulation",
        action="store_true",
        help="Recompute simulations even when a compatible cache exists.",
    )
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Processes used to fit independent amplitude traces.",
    )
    parser.add_argument(
        "--fit-method",
        choices=["direct", "gaussian"],
        default="direct",
        help="FWHM extraction method (direct is robust and much faster).",
    )
    parser.add_argument(
        "--cached-sweeps-only",
        action="store_true",
        help="Render simulation sweep heatmaps from cache without reading raw experiment data.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.simulation_steps_per_us < 1:
        raise ValueError("simulation_steps_per_us must be positive.")
    if args.minimum_steps_per_half < 1:
        raise ValueError("minimum_steps_per_half must be positive.")
    if args.fit_workers < 1:
        raise ValueError("fit_workers must be positive.")


def load_manifest(campaign_dir: Path) -> dict[str, Any]:
    path = campaign_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Campaign manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("pulse_shape") != "root_lorentzian":
        raise ValueError(
            "The matched qutrit comparison currently supports root_lorentzian "
            f"campaigns, not {manifest.get('pulse_shape')!r}."
        )
    if not manifest.get("echo", False):
        raise ValueError("The matched comparison expects an echo=True campaign.")
    if manifest.get("ac_stark_correction", False):
        raise ValueError(
            "The matched simulator does not include AC-Stark correction; analyze "
            "a campaign acquired with ac_stark_correction=False."
        )
    missing_device = {
        "anharmonicity_hz",
        "t1_s",
        "t2_star_s",
        "x180_amplitude_v",
        "x180_length_ns",
    } - set(manifest.get("device", {}))
    if missing_device:
        raise ValueError(
            "Campaign manifest is missing simulation device parameters: "
            + ", ".join(sorted(missing_device))
        )
    return manifest


def select_results_path(run_directory: Path, selection: str) -> Path:
    raw_path = run_directory / "results.npz"
    mitigated_path = run_directory / "results_mitigated.npz"
    if selection == "raw":
        selected = raw_path
    elif selection == "mitigated":
        selected = mitigated_path
    else:
        selected = mitigated_path if mitigated_path.is_file() else raw_path
    if not selected.is_file():
        raise FileNotFoundError(f"Saved result file does not exist: {selected}")
    return selected


def load_experimental_dataset(run_directory: Path, selection: str) -> xr.Dataset:
    sweep_path = run_directory / "sweep.npz"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"Saved sweep does not exist: {sweep_path}")
    with np.load(sweep_path, allow_pickle=False) as saved:
        qubits = np.asarray(saved["qubit"])
        detuning = np.asarray(saved["detuning"], dtype=float)
        amp_prefactor = np.asarray(saved["amp_prefactor"], dtype=float)
    results_path = select_results_path(run_directory, selection)
    with np.load(results_path, allow_pickle=False) as saved:
        if "state" not in saved.files:
            raise ValueError(f"State-discrimination data is missing from {results_path}")
        state = np.asarray(saved["state"], dtype=float)
    expected_shape = (qubits.size, detuning.size, amp_prefactor.size)
    if state.shape != expected_shape:
        raise ValueError(
            f"Unexpected state shape {state.shape} in {results_path}; "
            f"expected {expected_shape}."
        )
    return xr.Dataset(
        {"state": (("qubit", "detuning", "amp_prefactor"), state)},
        coords={
            "qubit": qubits,
            "detuning": detuning,
            "amp_prefactor": amp_prefactor,
        },
    )


def _fit_fwhm_chunk(dataset: xr.Dataset) -> xr.Dataset:
    fitted = add_gaussian_fwhm_analysis(
        dataset,
        use_state_discrimination=True,
    )
    return xr.Dataset(
        {
            "fwhm_hz": fitted.gaussian_fwhm_hz,
            "fit_score": fitted.gaussian_fit_r_squared,
        }
    )


def _interpolated_crossing(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    target: float,
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def _direct_trace_fwhm(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 7 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan, np.nan

    from scipy.signal import savgol_filter

    window = min(9, x.size if x.size % 2 else x.size - 1)
    smooth = savgol_filter(y, window_length=window, polyorder=min(2, window - 1))
    edge_count = max(3, x.size // 10)
    edge_indices = np.r_[0:edge_count, x.size - edge_count : x.size]
    baseline_coefficients = np.polyfit(x[edge_indices], smooth[edge_indices], 1)
    residual = smooth - np.polyval(baseline_coefficients, x)
    feature_index = int(np.argmax(np.abs(residual)))
    feature_height = float(residual[feature_index])
    sign = 1.0 if feature_height >= 0 else -1.0
    feature = sign * residual
    half_height = 0.5 * abs(feature_height)

    differences = np.diff(y)
    median_difference = float(np.median(differences))
    noise = 1.4826 * float(np.median(np.abs(differences - median_difference)))
    noise /= np.sqrt(2.0)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.finfo(float).eps
    signal_to_noise = abs(feature_height) / noise
    if signal_to_noise < 3.0 or feature_index in {0, x.size - 1}:
        return np.nan, signal_to_noise

    left_candidates = np.flatnonzero(feature[:feature_index] <= half_height)
    right_candidates = np.flatnonzero(feature[feature_index + 1 :] <= half_height)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return np.nan, signal_to_noise
    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = feature_index + 1 + int(right_candidates[0])
    right_low = right_high - 1
    left = _interpolated_crossing(
        x[left_low],
        x[left_high],
        feature[left_low],
        feature[left_high],
        half_height,
    )
    right = _interpolated_crossing(
        x[right_low],
        x[right_high],
        feature[right_low],
        feature[right_high],
        half_height,
    )
    width = float(right - left)
    if not np.isfinite(width) or width <= 0:
        return np.nan, signal_to_noise
    return width, signal_to_noise


def fit_direct_fwhm(dataset: xr.Dataset) -> xr.Dataset:
    detuning = np.asarray(dataset.detuning.values, dtype=float)
    qubits = np.asarray(dataset.qubit.values)
    amplitudes = np.asarray(dataset.amp_prefactor.values, dtype=float)
    widths = np.full((qubits.size, amplitudes.size), np.nan, dtype=float)
    scores = np.full_like(widths, np.nan)
    for qubit_index in range(qubits.size):
        for amplitude_index in range(amplitudes.size):
            widths[qubit_index, amplitude_index], scores[
                qubit_index, amplitude_index
            ] = _direct_trace_fwhm(
                detuning,
                np.asarray(
                    dataset.state.values[qubit_index, :, amplitude_index],
                    dtype=float,
                ),
            )
    return xr.Dataset(
        {
            "fwhm_hz": (("qubit", "amp_prefactor"), widths),
            "fit_score": (("qubit", "amp_prefactor"), scores),
        },
        coords={"qubit": qubits, "amp_prefactor": amplitudes},
    )


def fit_fwhm(dataset: xr.Dataset, *, workers: int = 1) -> xr.Dataset:
    """Fit independent amplitude traces in parallel while preserving the model."""
    amplitude_count = int(dataset.sizes.get("amp_prefactor", 0))
    worker_count = min(int(workers), amplitude_count)
    if worker_count <= 1:
        return _fit_fwhm_chunk(dataset)
    # Submit one trace at a time. Noisy traces can take far longer than clean
    # ones, so dynamic scheduling avoids leaving most workers idle behind one
    # pathological contiguous amplitude block.
    chunks = [
        dataset.isel(amp_prefactor=slice(index, index + 1))
        for index in range(amplitude_count)
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        fitted_chunks = list(executor.map(_fit_fwhm_chunk, chunks))
    return xr.concat(fitted_chunks, dim="amp_prefactor").sortby("amp_prefactor")


def extract_fwhm(
    dataset: xr.Dataset,
    *,
    method: str,
    workers: int,
) -> xr.Dataset:
    if method == "direct":
        return fit_direct_fwhm(dataset)
    if method == "gaussian":
        return fit_fwhm(dataset, workers=workers)
    raise ValueError(f"Unknown FWHM method: {method!r}")


def fitted_cache_path(
    output_dir: Path,
    *,
    source: str,
    pulse_length_us: float,
) -> Path:
    label = f"{pulse_length_us:g}".replace(".", "p")
    return output_dir / "fit_cache" / f"{source}_{label}us.npz"


def load_fitted_cache(path: Path, amp_prefactor: np.ndarray) -> xr.Dataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["amp_prefactor"], amp_prefactor):
            return None
        fwhm_hz = np.asarray(saved["fwhm_hz"], dtype=float)
        fit_score = np.asarray(saved["fit_score"], dtype=float)
    return xr.Dataset(
        {
            "fwhm_hz": (("qubit", "amp_prefactor"), fwhm_hz),
            "fit_score": (
                ("qubit", "amp_prefactor"),
                fit_score,
            ),
        },
        coords={"qubit": ["cached"], "amp_prefactor": amp_prefactor},
    )


def fit_fwhm_cached(
    dataset: xr.Dataset,
    *,
    workers: int,
    method: str,
    path: Path,
    force: bool = False,
) -> xr.Dataset:
    amp_prefactor = np.asarray(dataset.amp_prefactor.values, dtype=float)
    if not force:
        cached = load_fitted_cache(path, amp_prefactor)
        if cached is not None:
            print(f"  using cached FWHM fit: {path.name}", flush=True)
            return cached
    fitted = extract_fwhm(dataset, method=method, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        amp_prefactor=amp_prefactor,
        fwhm_hz=np.asarray(fitted.fwhm_hz.values, dtype=float),
        fit_score=np.asarray(
            fitted.fit_score.values,
            dtype=float,
        ),
    )
    return fitted


def simulation_steps(
    pulse_length_us: float,
    *,
    steps_per_us: int,
    minimum_steps: int,
) -> int:
    return max(minimum_steps, int(round(0.5 * pulse_length_us * steps_per_us)))


def simulation_cache_path(output_dir: Path, pulse_length_us: float) -> Path:
    label = f"{pulse_length_us:g}".replace(".", "p")
    return output_dir / "simulation_cache" / f"simulation_{label}us.npz"


def load_compatible_cache(
    path: Path,
    *,
    detuning_hz: np.ndarray,
    rabi_mhz: np.ndarray,
    pulse_length_us: float,
    num_steps_per_half: int,
) -> xr.Dataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        compatible = (
            float(saved["pulse_length_us"]) == pulse_length_us
            and int(saved["num_steps_per_half"]) == num_steps_per_half
            and np.array_equal(saved["detuning_hz"], detuning_hz)
            and np.allclose(saved["rabi_mhz"], rabi_mhz)
        )
        if not compatible:
            return None
        state = np.asarray(saved["total_excited_probability"], dtype=float)
        amp_prefactor = np.asarray(saved["amp_prefactor"], dtype=float)
    return xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                state.T[np.newaxis, ...],
            )
        },
        coords={
            "qubit": ["simulation"],
            "detuning": detuning_hz,
            "amp_prefactor": amp_prefactor,
        },
    )


def load_simulation_cache(path: Path) -> xr.Dataset:
    """Load a saved simulation grid without requiring the experimental source."""
    if not path.is_file():
        raise FileNotFoundError(f"Simulation cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as saved:
        detuning_hz = np.asarray(saved["detuning_hz"], dtype=float)
        amp_prefactor = np.asarray(saved["amp_prefactor"], dtype=float)
        state = np.asarray(saved["total_excited_probability"], dtype=float)
    return xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                state.T[np.newaxis, ...],
            )
        },
        coords={
            "qubit": ["simulation"],
            "detuning": detuning_hz,
            "amp_prefactor": amp_prefactor,
        },
    )


def simulate_dataset(
    manifest: dict[str, Any],
    experimental: xr.Dataset,
    *,
    pulse_length_us: float,
    output_dir: Path,
    steps_per_us: int,
    minimum_steps: int,
    force: bool,
) -> xr.Dataset:
    from qutrit_slices import simulate_qutrit_slices

    device = manifest["device"]
    detuning_hz = np.asarray(experimental.detuning.values, dtype=float)
    amp_prefactor = np.asarray(experimental.amp_prefactor.values, dtype=float)
    full_amplitude_v = amp_prefactor * float(manifest["max_amplitude_v"])
    rabi_hz = amplitude_to_rabi_frequency_hz(
        full_amplitude_v,
        float(device["x180_amplitude_v"]),
        float(device["x180_length_ns"]),
    )
    rabi_mhz = np.asarray(rabi_hz, dtype=float) / 1e6
    num_steps = simulation_steps(
        pulse_length_us,
        steps_per_us=steps_per_us,
        minimum_steps=minimum_steps,
    )
    cache_path = simulation_cache_path(output_dir, pulse_length_us)
    if not force:
        cached = load_compatible_cache(
            cache_path,
            detuning_hz=detuning_hz,
            rabi_mhz=rabi_mhz,
            pulse_length_us=pulse_length_us,
            num_steps_per_half=num_steps,
        )
        if cached is not None:
            print(f"  using cached simulation: {cache_path.name}", flush=True)
            return cached

    result = simulate_qutrit_slices(
        duration_us=pulse_length_us,
        detuning_mhz=detuning_hz / 1e6,
        rabi_mhz=rabi_mhz,
        t1_us=float(device["t1_s"]) * 1e6,
        t2_star_us=float(device["t2_star_s"]) * 1e6,
        anharmonicity_mhz=-abs(float(device["anharmonicity_hz"])) / 1e6,
        num_steps_per_half=num_steps,
        cutoff=float(manifest["cutoff"]),
        echo=True,
    )
    total_excited = result.excited + result.second_excited
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        pulse_length_us=pulse_length_us,
        num_steps_per_half=num_steps,
        detuning_hz=detuning_hz,
        amp_prefactor=amp_prefactor,
        rabi_mhz=rabi_mhz,
        total_excited_probability=total_excited,
    )
    return xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                total_excited.T[np.newaxis, ...],
            )
        },
        coords={
            "qubit": ["simulation"],
            "detuning": detuning_hz,
            "amp_prefactor": amp_prefactor,
        },
    )


def comparison_records(
    manifest: dict[str, Any],
    *,
    pulse_length_us: float,
    experimental: xr.Dataset,
    simulated: xr.Dataset,
) -> list[dict[str, Any]]:
    device = manifest["device"]
    t2_star_s = float(device["t2_star_s"])
    t2_limit_hz = 1.0 / (np.pi * t2_star_s)
    amps = np.asarray(experimental.amp_prefactor.values, dtype=float)
    amplitude_v = amps * float(manifest["max_amplitude_v"])
    rabi_mhz = np.asarray(
        amplitude_to_rabi_frequency_hz(
            amplitude_v,
            float(device["x180_amplitude_v"]),
            float(device["x180_length_ns"]),
        ),
        dtype=float,
    ) / 1e6
    exp = experimental.isel(qubit=0)
    sim = simulated.isel(qubit=0)
    records = []
    for index in range(amps.size):
        measured_hz = float(exp.fwhm_hz.values[index])
        expected_hz = float(sim.fwhm_hz.values[index])
        records.append(
            {
                "pulse_length_us": pulse_length_us,
                "amp_prefactor": amps[index],
                "amplitude_v": amplitude_v[index],
                "rabi_frequency_mhz": rabi_mhz[index],
                "measured_fwhm_hz": measured_hz,
                "simulated_fwhm_hz": expected_hz,
                "difference_hz": measured_hz - expected_hz,
                "t2_star_s": t2_star_s,
                "t2_limit_hz": t2_limit_hz,
                "measured_fwhm_t2_units": measured_hz / t2_limit_hz,
                "simulated_fwhm_t2_units": expected_hz / t2_limit_hz,
                "difference_t2_units": (measured_hz - expected_hz) / t2_limit_hz,
                "ratio_measured_to_simulated": (
                    measured_hz / expected_hz if expected_hz > 0 else np.nan
                ),
                "measured_fit_score": float(exp.fit_score.values[index]),
                "simulated_fit_score": float(sim.fit_score.values[index]),
            }
        )
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_overlays(records: list[dict[str, Any]], output_dir: Path) -> Path:
    lengths = sorted({float(record["pulse_length_us"]) for record in records})
    columns = 2
    rows = int(np.ceil(len(lengths) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 2.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, length in zip(axes, lengths, strict=False):
        selected = [record for record in records if record["pulse_length_us"] == length]
        amplitude = np.asarray([record["amplitude_v"] for record in selected])
        measured = np.asarray(
            [record["measured_fwhm_t2_units"] for record in selected]
        )
        simulated = np.asarray(
            [record["simulated_fwhm_t2_units"] for record in selected]
        )
        axis.plot(amplitude, simulated, color="#d95f02", linewidth=1.5, label="Simulation")
        axis.scatter(amplitude, measured, color="#1b6ca8", s=13, label="Experiment", zorder=3)
        axis.axhline(
            1.0,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label=r"$T_2^*$ limit",
            zorder=2,
        )
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.set_title(f"{length:g} us")
        axis.set_xlabel("Peak amplitude (V)")
        axis.set_ylabel(r"FWHM / $[1/(\pi T_2^*)]$")
    for axis in axes[len(lengths):]:
        axis.set_visible(False)
    if lengths:
        axes[0].legend(frameon=False)
    figure.suptitle("Echo shaped-pulse linewidth: experiment vs simulation")
    path = output_dir / "fwhm_experiment_vs_simulation_by_length.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_parity(records: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    lengths = sorted({float(record["pulse_length_us"]) for record in records})
    color_map = plt.get_cmap("viridis")
    finite_values: list[float] = []
    for index, length in enumerate(lengths):
        selected = [record for record in records if record["pulse_length_us"] == length]
        measured = np.asarray(
            [record["measured_fwhm_t2_units"] for record in selected]
        )
        simulated = np.asarray(
            [record["simulated_fwhm_t2_units"] for record in selected]
        )
        valid = np.isfinite(measured) & np.isfinite(simulated)
        finite_values.extend(measured[valid].tolist())
        finite_values.extend(simulated[valid].tolist())
        axis.scatter(
            simulated[valid],
            measured[valid],
            s=14,
            alpha=0.75,
            color=color_map(index / max(1, len(lengths) - 1)),
            label=f"{length:g} us",
        )
    if finite_values:
        lower = min(finite_values)
        upper = max(finite_values)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="1:1")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    axis.set_xlabel(r"Simulated FWHM / $[1/(\pi T_2^*)]$")
    axis.set_ylabel(r"Measured FWHM / $[1/(\pi T_2^*)]$")
    axis.set_title("Measured versus simulated linewidth")
    axis.grid(alpha=0.25)
    axis.legend(ncols=2, fontsize=8, frameon=False)
    path = output_dir / "fwhm_measured_vs_simulated_parity.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_simulation_sweep(
    dataset: xr.Dataset,
    manifest: dict[str, Any],
    *,
    pulse_length_us: float,
    index: int,
    output_dir: Path,
) -> Path:
    """Save a detuning-by-peak-amplitude heatmap for the matched simulation."""
    detuning_mhz = np.asarray(dataset.detuning.values, dtype=float) / 1e6
    amplitude_v = (
        np.asarray(dataset.amp_prefactor.values, dtype=float)
        * float(manifest["max_amplitude_v"])
    )
    device = manifest["device"]
    rabi_mhz = np.asarray(
        amplitude_to_rabi_frequency_hz(
            amplitude_v,
            float(device["x180_amplitude_v"]),
            float(device["x180_length_ns"]),
        ),
        dtype=float,
    ) / 1e6
    probability = np.asarray(dataset.state.isel(qubit=0).values, dtype=float).T
    # Match the standard experiment figure geometry exactly.  This includes
    # the calibration parameter footer and the deliberately wide colorbar gap,
    # so side-by-side HTML panels have equal heatmap rectangles rather than
    # merely equal outer image widths.
    figure, axis = plt.subplots(figsize=(13, 8))
    image = axis.pcolormesh(
        detuning_mhz,
        rabi_mhz,
        probability,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel("Detuning [MHz]")
    axis.set_ylabel("Rabi frequency [MHz]")
    t2_half_width_mhz = 1.0 / (
        2.0 * np.pi * float(device["t2_star_s"]) * 1e6
    )
    axis.axvline(-t2_half_width_mhz, color="white", linestyle="--", linewidth=1.0)
    axis.axvline(t2_half_width_mhz, color="white", linestyle="--", linewidth=1.0)

    x180_rabi_mhz = 1e3 / (2.0 * float(device["x180_length_ns"]))
    x180_amplitude_v = float(device["x180_amplitude_v"])
    amplitude_axis = axis.secondary_yaxis(
        "right",
        functions=(
            lambda frequency_mhz: frequency_mhz / x180_rabi_mhz * x180_amplitude_v,
            lambda voltage: voltage / x180_amplitude_v * x180_rabi_mhz,
        ),
    )
    amplitude_axis.set_ylabel("Lorentzian peak amplitude [V]")

    rf_frequency_hz = float(device["rf_frequency_hz"])
    frequency_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda offset_mhz: (rf_frequency_hz + offset_mhz * 1e6) / 1e9,
            lambda frequency_ghz: (frequency_ghz * 1e9 - rf_frequency_hz) / 1e6,
        ),
    )
    frequency_axis.set_xlabel("RF frequency [GHz]")
    axis.set_title(
        f"{manifest['qubit']}: simulated state"
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.16)
    colorbar.set_label(r"Final $P(|1\rangle)+P(|2\rangle)$")
    minimum_factor = float(manifest["min_amplitude_v"]) / float(
        manifest["max_amplitude_v"]
    )
    parameter_lines = [
        "Parameters",
        (
            f"pulse shape={manifest['pulse_shape']}, echo={manifest['echo']}, "
            f"pulse length={pulse_length_us * 1000:g} ns, "
            f"template length={manifest['template_length_ns']} ns"
        ),
        (
            f"peak amp={float(manifest['max_amplitude_v']) * 1000:g} mV, "
            f"{float(manifest['cutoff']):g} cutoff"
        ),
        (
            f"amp factor={minimum_factor:g}:log:{1:g}, detuning span="
            f"{float(manifest['frequency_span_mhz']):g} MHz"
        ),
        (
            f"{manifest['qubit']}: RF={rf_frequency_hz / 1e9:.6f} GHz | "
            f"x180 square pi: amp={x180_amplitude_v * 1000:.3f} mV, "
            f"t_pi={float(device['x180_length_ns']):g} ns | "
            f"T1={float(device['t1_s']) * 1e6:.4f} us | "
            f"T2*={float(device['t2_star_s']) * 1e6:.3f} us"
        ),
    ]
    figure.text(
        0.01,
        0.01,
        "\n".join(parameter_lines),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )
    figure.subplots_adjust(top=0.95, bottom=0.25, right=0.86)
    label = f"{pulse_length_us:g}".replace(".", "p")
    path = output_dir / f"simulation_sweep_{index:02d}_{label}us.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def analyze(args: argparse.Namespace) -> list[Path]:
    campaign_dir = args.campaign_dir.resolve()
    manifest = load_manifest(campaign_dir)
    output_dir = (args.output_dir or campaign_dir / "fwhm_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    successful_runs = [
        record
        for record in manifest.get("runs", [])
        if record.get("status") == "ok" and record.get("run_directory")
    ]
    if not successful_runs:
        raise ValueError("The campaign manifest contains no successful hardware runs.")

    if args.cached_sweeps_only:
        paths: list[Path] = []
        for index, run in enumerate(successful_runs, start=1):
            pulse_length_us = float(run["pulse_length_us"])
            cached = load_simulation_cache(
                simulation_cache_path(output_dir, pulse_length_us)
            )
            paths.append(
                plot_simulation_sweep(
                    cached,
                    manifest,
                    pulse_length_us=pulse_length_us,
                    index=index,
                    output_dir=output_dir,
                )
            )
        return paths

    records: list[dict[str, Any]] = []
    simulation_plot_paths: list[Path] = []
    for index, run in enumerate(successful_runs, start=1):
        pulse_length_us = float(run["pulse_length_us"])
        print(
            f"[{index}/{len(successful_runs)}] Fitting and simulating "
            f"{pulse_length_us:g} us...",
            flush=True,
        )
        experimental_raw = load_experimental_dataset(
            Path(run["run_directory"]),
            args.results,
        )
        experimental = fit_fwhm_cached(
            experimental_raw,
            workers=args.fit_workers,
            method=args.fit_method,
            path=fitted_cache_path(
                output_dir,
                source=f"experiment_{args.fit_method}_{args.results}",
                pulse_length_us=pulse_length_us,
            ),
        )
        simulated_raw = simulate_dataset(
            manifest,
            experimental_raw,
            pulse_length_us=pulse_length_us,
            output_dir=output_dir,
            steps_per_us=args.simulation_steps_per_us,
            minimum_steps=args.minimum_steps_per_half,
            force=args.force_simulation,
        )
        simulation_plot_paths.append(
            plot_simulation_sweep(
                simulated_raw,
                manifest,
                pulse_length_us=pulse_length_us,
                index=index,
                output_dir=output_dir,
            )
        )
        simulated = fit_fwhm_cached(
            simulated_raw,
            workers=args.fit_workers,
            method=args.fit_method,
            path=fitted_cache_path(
                output_dir,
                source=(
                    f"simulation_{args.fit_method}_{args.simulation_steps_per_us}"
                ),
                pulse_length_us=pulse_length_us,
            ),
            force=args.force_simulation,
        )
        records.extend(
            comparison_records(
                manifest,
                pulse_length_us=pulse_length_us,
                experimental=experimental,
                simulated=simulated,
            )
        )

    csv_path = output_dir / "fwhm_experiment_vs_simulation.csv"
    write_csv(csv_path, records)
    paths = [
        csv_path,
        *simulation_plot_paths,
        plot_overlays(records, output_dir),
        plot_parity(records, output_dir),
    ]
    summary = {
        "campaign_directory": str(campaign_dir),
        "results_selection": args.results,
        "simulation_steps_per_us": args.simulation_steps_per_us,
        "minimum_steps_per_half": args.minimum_steps_per_half,
        "fit_workers": args.fit_workers,
        "fit_method": args.fit_method,
        "t2_star_s": float(manifest["device"]["t2_star_s"]),
        "t2_limit_hz": 1.0 / (
            np.pi * float(manifest["device"]["t2_star_s"])
        ),
        "successful_lengths": len(successful_runs),
        "comparison_points": len(records),
        "finite_comparisons": int(
            sum(
                bool(np.isfinite(record["measured_fwhm_hz"]))
                and bool(np.isfinite(record["simulated_fwhm_hz"]))
                for record in records
            )
        ),
        "outputs": [str(path) for path in paths],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths.append(summary_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    for path in analyze(args):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
