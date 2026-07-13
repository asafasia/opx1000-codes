"""Run a PC-only cutoff-amplitude-FWHM map with the shared data fitter."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr

from parameters import SimulationParameters
from simulate_echo_lorentzian import (
    plot_dataset,
    serializable_parameters,
    simulate_grid,
)


def cutoff_values(minimum: float, maximum: float, points: int) -> np.ndarray:
    if not 0 < minimum <= maximum <= 1:
        raise ValueError("Cutoffs must satisfy 0 < minimum <= maximum <= 1.")
    if points < 2:
        raise ValueError("cutoff_points must be at least 2.")
    return np.geomspace(maximum, minimum, points)


def records_from_dataset(
    ds: xr.Dataset,
    *,
    cutoff: float,
    run_index: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = ds.isel(qubit=0)
    for amp_index, amp_prefactor in enumerate(ds.amp_prefactor.values):
        point = selected.isel(amp_prefactor=amp_index)
        fwhm_hz = _scalar(point, "gaussian_fwhm_hz")
        records.append(
            {
                "run_index": run_index,
                "cutoff": float(cutoff),
                "qubit": str(ds.qubit.values[0]),
                "amp_prefactor": float(amp_prefactor),
                "full_amp_v": _scalar(point, "full_amp", coordinate=True),
                "rabi_frequency_mhz": _scalar(
                    point,
                    "rabi_frequency_hz",
                    coordinate=True,
                )
                / 1e6,
                "gaussian_center_hz": _scalar(point, "gaussian_center_hz"),
                "fwhm_hz": fwhm_hz,
                "fwhm_mhz": fwhm_hz / 1e6,
                "t2_star_s": _scalar(point, "t2_star_s", coordinate=True),
                "t2_star_fwhm_limit_hz": _scalar(
                    point,
                    "t2_star_fwhm_limit_hz",
                    coordinate=True,
                ),
                "fwhm_t2_star_units": _scalar(
                    point,
                    "gaussian_fwhm_t2_star_units",
                ),
                "fit_amplitude": _scalar(point, "gaussian_fit_amplitude"),
                "fit_abs_amplitude": _scalar(
                    point,
                    "gaussian_fit_abs_amplitude",
                ),
                "fit_r_squared": _scalar(point, "gaussian_fit_r_squared"),
                "fit_model": _string_scalar(point, "gaussian_fit_model"),
                "positive_fwhm_hz": _scalar(
                    point,
                    "gaussian_positive_fwhm_hz",
                ),
                "positive_fit_abs_amplitude": _scalar(
                    point,
                    "gaussian_positive_fit_abs_amplitude",
                ),
                "positive_fit_r_squared": _scalar(
                    point,
                    "gaussian_positive_fit_r_squared",
                ),
                "negative_fwhm_hz": _scalar(
                    point,
                    "gaussian_negative_fwhm_hz",
                ),
                "negative_fit_abs_amplitude": _scalar(
                    point,
                    "gaussian_negative_fit_abs_amplitude",
                ),
                "negative_fit_r_squared": _scalar(
                    point,
                    "gaussian_negative_fit_r_squared",
                ),
            }
        )
    return records


def best_signal_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        record
        for record in records
        if _finite(record["fwhm_hz"]) and _finite(record["fit_abs_amplitude"])
    ]
    return max(valid, key=lambda record: record["fit_abs_amplitude"]) if valid else None


def run_cutoff_map(
    parameters: SimulationParameters,
    cutoffs: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "individual_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    interrupted = False
    started_at = datetime.now().astimezone()
    started_s = time.perf_counter()

    try:
        for run_index, cutoff in enumerate(cutoffs, start=1):
            print(
                f"Cutoff simulation {run_index}/{len(cutoffs)}: {float(cutoff):.6g}",
                flush=True,
            )
            point_parameters = replace(parameters, cutoff=float(cutoff))
            ds = simulate_grid(point_parameters)
            run_records = records_from_dataset(
                ds,
                cutoff=float(cutoff),
                run_index=run_index,
            )
            records.extend(run_records)
            best = best_signal_record(run_records)
            if best is not None:
                best_records.append(best)
            figure_name = f"cutoff_{run_index:02d}_{float(cutoff):.3e}.png"
            plot_dataset(
                ds,
                figures_dir,
                filename=figure_name,
                title=f"q1 echo simulation: cutoff={float(cutoff):.3g}",
            )
            runs.append(
                {
                    "run_index": run_index,
                    "cutoff": float(cutoff),
                    "valid_fits": int(np.isfinite(ds.gaussian_fwhm_hz).sum()),
                }
            )
            _write_progress_checkpoint(
                output_dir,
                parameters,
                records,
                best_records,
                runs,
                started_at=started_at,
                started_s=started_s,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Cutoff simulation interrupted; saving completed results.")

    duration_s = time.perf_counter() - started_s
    _write_csv(output_dir / "cutoff_amp_fwhm_map_fit_results.csv", records)
    _write_csv(output_dir / "cutoff_amp_fwhm_map_best_signal.csv", best_records)
    plot_cutoff_summary(best_records, output_dir)
    plot_fwhm_heatmap(records, output_dir)
    plot_per_cutoff_traces(records, output_dir)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "started_at": started_at.isoformat(),
        "duration_s": duration_s,
        "interrupted": interrupted,
        "completed_runs": len(runs),
        "runs": runs,
        "parameters": serializable_parameters(parameters),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved cutoff map to {output_dir}")
    print(f"Duration: {duration_s:.1f} s")
    return {
        "records": records,
        "best_records": best_records,
        "interrupted": interrupted,
        "duration_s": duration_s,
    }


def _write_progress_checkpoint(
    output_dir: Path,
    parameters: SimulationParameters,
    records: list[dict[str, Any]],
    best_records: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    started_at: datetime,
    started_s: float,
) -> None:
    """Persist completed cutoff results so a long sweep can be resumed manually."""
    _write_csv(output_dir / "cutoff_amp_fwhm_map_fit_results.csv", records)
    _write_csv(output_dir / "cutoff_amp_fwhm_map_best_signal.csv", best_records)
    progress = {
        "status": "running",
        "updated_at": datetime.now().astimezone().isoformat(),
        "started_at": started_at.isoformat(),
        "duration_s": time.perf_counter() - started_s,
        "completed_runs": len(runs),
        "runs": runs,
        "parameters": serializable_parameters(parameters),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(progress, indent=2),
        encoding="utf-8",
    )


def plot_cutoff_summary(records: list[dict[str, Any]], output_dir: Path) -> Path | None:
    if not records:
        return None
    ordered = sorted(records, key=lambda record: record["cutoff"])
    cutoffs = [record["cutoff"] for record in ordered]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        cutoffs,
        [record["fwhm_t2_star_units"] for record in ordered],
        "o-",
        label=ordered[0]["qubit"],
    )
    axes[1].plot(
        cutoffs,
        [record["fit_abs_amplitude"] for record in ordered],
        "o-",
        label=ordered[0]["qubit"],
    )
    axes[0].set_ylabel("FWHM / (1/(pi*T2*))")
    axes[1].set_ylabel("Fit signal amplitude")
    axes[1].set_xlabel("Cutoff")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    path = output_dir / "cutoff_amp_fwhm_map_summary.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_fwhm_heatmap(records: list[dict[str, Any]], output_dir: Path) -> Path | None:
    valid = [
        record
        for record in records
        if _finite(record["fwhm_t2_star_units"])
        and record["fwhm_t2_star_units"] > 0
        and _finite(record["full_amp_v"])
        and record["full_amp_v"] > 0
    ]
    if not valid:
        return None
    resolution_cmap = plt.get_cmap("viridis").with_extremes(bad="white")
    score_cmap = plt.get_cmap("magma").with_extremes(bad="white")
    cutoffs = np.array(sorted({record["cutoff"] for record in valid}))
    amplitudes_v = np.array(sorted({record["full_amp_v"] for record in valid}))
    resolution = np.full((len(amplitudes_v), len(cutoffs)), np.nan)
    resolution_times_signal = np.full_like(resolution, np.nan)
    cutoff_index = {value: index for index, value in enumerate(cutoffs)}
    amplitude_index = {value: index for index, value in enumerate(amplitudes_v)}
    for record in valid:
        row = amplitude_index[record["full_amp_v"]]
        column = cutoff_index[record["cutoff"]]
        point_resolution = 1 / record["fwhm_t2_star_units"]
        resolution[row, column] = point_resolution
        if _finite(record["fit_abs_amplitude"]) and record["fit_abs_amplitude"] > 0:
            resolution_times_signal[row, column] = (
                point_resolution * record["fit_abs_amplitude"]
            )
    figure, axes = plt.subplots(2, 1, figsize=(8, 14))
    for axis, values, label, title, cmap in (
        (
            axes[0],
            resolution,
            "Resolution: (1/(pi*T2*)) / FWHM",
            f"{valid[0]['qubit']}: reciprocal FWHM resolution",
            resolution_cmap,
        ),
        (
            axes[1],
            resolution_times_signal,
            "Resolution * fit signal",
            f"{valid[0]['qubit']}: resolution multiplied by fit signal",
            score_cmap,
        ),
    ):
        image_options = (
            {"vmin": 0.1, "vmax": 1}
            if axis is axes[0]
            else {"norm": LogNorm(vmin=1e-1, vmax=1, clip=True)}
        )
        image = axis.pcolormesh(
            _cell_edges_log(cutoffs),
            _cell_edges_log(amplitudes_v),
            values,
            shading="auto",
            cmap=cmap,
            **image_options,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Cutoff")
        axis.set_ylabel("Full pulse amplitude [V]")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    path = output_dir / "cutoff_amp_fwhm_map_fwhm_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _cell_edges_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] / np.sqrt(10), values[0] * np.sqrt(10)])
    log_values = np.log10(values)
    midpoints = (log_values[:-1] + log_values[1:]) / 2
    first = log_values[0] - (midpoints[0] - log_values[0])
    last = log_values[-1] + (log_values[-1] - midpoints[-1])
    return 10 ** np.concatenate(([first], midpoints, [last]))


def plot_per_cutoff_traces(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    if not records:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    cutoffs = sorted({record["cutoff"] for record in records})
    colors = plt.cm.viridis(np.linspace(0, 1, len(cutoffs)))
    for cutoff, color in zip(cutoffs, colors):
        selected = sorted(
            (record for record in records if record["cutoff"] == cutoff),
            key=lambda record: record["rabi_frequency_mhz"],
        )
        x = [record["rabi_frequency_mhz"] for record in selected]
        axes[0].plot(
            x,
            [record["fwhm_t2_star_units"] for record in selected],
            "o-",
            color=color,
            markersize=3,
            label=f"{cutoff:.3g}",
        )
        axes[1].plot(
            x,
            [record["fit_abs_amplitude"] for record in selected],
            "o-",
            color=color,
            markersize=3,
            label=f"{cutoff:.3g}",
        )
    axes[0].set_ylabel("FWHM / (1/(pi*T2*))")
    axes[1].set_ylabel("Fit signal amplitude")
    axes[1].set_xlabel("Rabi frequency [MHz]")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(title="cutoff", fontsize=7, ncols=2)
    figure.tight_layout()
    path = output_dir / "cutoff_amp_fwhm_map_per_cutoff_traces.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _scalar(point: xr.Dataset, name: str, *, coordinate: bool = False) -> float:
    container = point.coords if coordinate else point
    if name not in container:
        return np.nan
    value = np.asarray(container[name].values, dtype=float)
    return float(value.reshape(-1)[0]) if value.size else np.nan


def _string_scalar(point: xr.Dataset, name: str) -> str:
    if name not in point:
        return ""
    return str(np.asarray(point[name].values).reshape(-1)[0])


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-min", type=float, default=0.002)
    parser.add_argument("--cutoff-max", type=float, default=0.99)
    parser.add_argument("--cutoff-points", type=int, default=10)
    parser.add_argument("--pulse-length-ns", type=int, default=20000)
    parser.add_argument("--peak-amplitude", type=float, default=0.15127819777954318)
    parser.add_argument("--min-amp-factor", type=float, default=0.01)
    parser.add_argument("--max-amp-factor", type=float, default=1.0)
    parser.add_argument("--amp-factor-points", type=int, default=16)
    parser.add_argument("--amp-factor-spacing", choices=["linear", "log"], default="log")
    parser.add_argument("--frequency-span-mhz", type=float, default=1.0)
    parser.add_argument("--frequency-points", type=int, default=81)
    parser.add_argument("--num-time-points", type=int, default=1000)
    parser.add_argument("--echo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulations/cutoff_amp_fwhm_map_20us_echo_q1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = SimulationParameters(
        pulse_shape="root_lorentzian",
        echo=args.echo,
        lorentzian_length_in_ns=args.pulse_length_ns,
        waveform_template_length_in_ns=args.pulse_length_ns,
        lorentzian_peak_amplitude=args.peak_amplitude,
        min_amp_factor=args.min_amp_factor,
        max_amp_factor=args.max_amp_factor,
        amp_factor_points=args.amp_factor_points,
        amp_factor_spacing=args.amp_factor_spacing,
        frequency_span_in_mhz=args.frequency_span_mhz,
        frequency_points=args.frequency_points,
        num_time_points=args.num_time_points,
        output_dir=args.output_dir,
    )
    run_cutoff_map(
        parameters,
        cutoff_values(args.cutoff_min, args.cutoff_max, args.cutoff_points),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
