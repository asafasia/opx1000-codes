"""Summarize FWHM RMS versus pulse length over a data-driven good-SNR band."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--snr-floor",
        type=float,
        default=3.0,
        help="Absolute minimum trace SNR inherited from the direct FWHM method.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.5,
        help="Adaptive threshold as this fraction of the smoothed-SNR 90th percentile.",
    )
    parser.add_argument(
        "--smoothing-points",
        type=int,
        default=9,
        help="Odd rolling-median width along the log-amplitude axis.",
    )
    parser.add_argument(
        "--max-gap-points",
        type=int,
        default=3,
        help="Maximum short candidate gap bridged when identifying a signal band.",
    )
    parser.add_argument(
        "--max-width-fraction",
        type=float,
        default=0.8,
        help="Reject FWHM values wider than this fraction of the detuning window.",
    )
    parser.add_argument(
        "--max-adjacent-width-ratio",
        type=float,
        default=2.5,
        help="Split bands at larger adjacent simulated-FWHM branch jumps.",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def read_comparisons(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: finite_float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    bridged = np.asarray(mask, dtype=bool).copy()
    true_indices = np.flatnonzero(bridged)
    for left, right in zip(true_indices[:-1], true_indices[1:], strict=False):
        if 0 < right - left - 1 <= max_gap:
            bridged[left : right + 1] = True
    return bridged


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts, stops, strict=True))


def select_good_snr_band(
    rows: list[dict[str, float]],
    *,
    snr_floor: float,
    relative_threshold: float,
    smoothing_points: int,
    max_gap_points: int,
    max_width_hz: float,
    max_adjacent_width_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    snr = np.asarray([row["measured_fit_score"] for row in rows])
    measured_hz = np.asarray([row["measured_fwhm_hz"] for row in rows])
    simulated_hz = np.asarray([row["simulated_fwhm_hz"] for row in rows])
    finite_snr = np.isfinite(snr)
    fill_value = float(np.nanmedian(snr[finite_snr])) if finite_snr.any() else 0.0
    filled_snr = np.where(finite_snr, snr, fill_value)
    window = max(3, int(smoothing_points) | 1)
    smoothed_snr = median_filter(filled_snr, size=window, mode="nearest")
    upper_quality = float(np.nanpercentile(smoothed_snr, 90))
    # The smoothed profile locates a contiguous band; individual accepted points
    # must still pass the stricter raw-SNR floor below.  A slightly lower band
    # floor prevents weak-but-coherent long-pulse features from fragmenting.
    band_floor = max(1.0, float(snr_floor) - 1.0)
    threshold = max(band_floor, float(relative_threshold) * upper_quality)

    resolved = (
        np.isfinite(measured_hz)
        & np.isfinite(simulated_hz)
        & np.isfinite(snr)
        & (snr >= snr_floor)
        & (measured_hz < max_width_hz)
        & (simulated_hz < max_width_hz)
    )
    candidate = resolved & (smoothed_snr >= threshold)
    support = bridge_short_gaps(candidate, max_gap_points)
    adjacent_ratio = np.maximum(
        simulated_hz[:-1] / np.maximum(simulated_hz[1:], np.finfo(float).eps),
        simulated_hz[1:] / np.maximum(simulated_hz[:-1], np.finfo(float).eps),
    )
    # A sharp simulated-width jump identifies a change of extraction branch,
    # not a continuous physical amplitude band.  Split after gap bridging so
    # the bridge cannot reconnect the two branches.
    for boundary in np.flatnonzero(adjacent_ratio > max_adjacent_width_ratio):
        support[int(boundary)] = False
    runs = contiguous_runs(support)
    if not runs:
        return np.zeros(len(rows), dtype=bool), {
            "snr_threshold": threshold,
            "upper_quality_snr": upper_quality,
        }

    def run_value(run: tuple[int, int]) -> float:
        start, stop = run
        count = int(np.count_nonzero(resolved[start:stop]))
        simulated_width = float(np.nanmedian(simulated_hz[start:stop]))
        # Prefer a long, scan-resolved physical band.  Dividing by simulated
        # linewidth rejects broad high-amplitude islands without selecting on
        # agreement between the measured and simulated FWHM values.
        return count / max(simulated_width, np.finfo(float).eps)

    start, stop = max(runs, key=run_value)
    selected = np.zeros(len(rows), dtype=bool)
    selected[start:stop] = resolved[start:stop]
    return selected, {
        "snr_threshold": threshold,
        "upper_quality_snr": upper_quality,
        "band_start_index": float(start),
        "band_stop_index": float(stop),
    }


def summarize_length(
    length: float,
    rows: list[dict[str, float]],
    *,
    detuning_span_hz: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    rows.sort(key=lambda row: row["amplitude_v"])
    selected, diagnostics = select_good_snr_band(
        rows,
        snr_floor=args.snr_floor,
        relative_threshold=args.relative_threshold,
        smoothing_points=args.smoothing_points,
        max_gap_points=args.max_gap_points,
        max_width_hz=args.max_width_fraction * detuning_span_hz,
        max_adjacent_width_ratio=args.max_adjacent_width_ratio,
    )
    amplitude = np.asarray([row["amplitude_v"] for row in rows])
    measured = np.asarray([row["measured_fwhm_t2_units"] for row in rows])
    simulated = np.asarray([row["simulated_fwhm_t2_units"] for row in rows])
    snr = np.asarray([row["measured_fit_score"] for row in rows])
    measured_good = measured[selected]
    simulated_good = simulated[selected]
    difference_good = measured_good - simulated_good
    count = int(np.count_nonzero(selected))
    result = {
        "pulse_length_us": length,
        "selected_points": count,
        "total_points": len(rows),
        "amplitude_min_v": float(np.min(amplitude[selected])) if count else np.nan,
        "amplitude_max_v": float(np.max(amplitude[selected])) if count else np.nan,
        "median_snr": float(np.median(snr[selected])) if count else np.nan,
        "accepted_snr_floor": args.snr_floor,
        "snr_threshold": diagnostics["snr_threshold"],
        "upper_quality_snr": diagnostics["upper_quality_snr"],
        "experiment_fwhm_rms_t2": (
            float(np.sqrt(np.mean(measured_good**2))) if count else np.nan
        ),
        "simulation_fwhm_rms_t2": (
            float(np.sqrt(np.mean(simulated_good**2))) if count else np.nan
        ),
        "experiment_simulation_rmse_t2": (
            float(np.sqrt(np.mean(difference_good**2))) if count else np.nan
        ),
        "median_absolute_error_t2": (
            float(np.median(np.abs(difference_good))) if count else np.nan
        ),
    }
    simulation_rms = result["simulation_fwhm_rms_t2"]
    result["relative_rmse"] = (
        result["experiment_simulation_rmse_t2"] / simulation_rms
        if np.isfinite(simulation_rms) and simulation_rms > 0
        else np.nan
    )
    return result, selected


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def plot_rms(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    t2_limit_hz: float,
) -> Path:
    length = np.asarray([row["pulse_length_us"] for row in summaries])
    t2_limit_khz = float(t2_limit_hz) / 1e3
    experiment_t2 = np.asarray([row["experiment_fwhm_rms_t2"] for row in summaries])
    simulation_t2 = np.asarray([row["simulation_fwhm_rms_t2"] for row in summaries])
    experiment = experiment_t2 * t2_limit_khz
    simulation = simulation_t2 * t2_limit_khz
    rmse = np.asarray([row["experiment_simulation_rmse_t2"] for row in summaries])
    relative = np.asarray([row["relative_rmse"] for row in summaries])

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(length, simulation, "o-", color="#d95f02", label="Simulation RMS")
    axes[0].plot(length, experiment, "o-", color="#1b6ca8", label="Experiment RMS")
    constant_path = output_dir / "constant_pulse_t2_limited_reference.csv"
    if constant_path.is_file():
        constant_rows = read_comparisons(constant_path)
        constant_length = np.asarray([row["pulse_length_us"] for row in constant_rows])
        constant_fwhm = (
            np.asarray([row["fwhm_t2_units"] for row in constant_rows])
            * t2_limit_khz
        )
        axes[0].plot(
            constant_length,
            constant_fwhm,
            "o-",
            color="#b23a48",
            linewidth=1.6,
            label="T2*-limited constant pulse",
        )
    axes[0].axhline(
        t2_limit_khz,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label=rf"$T_2^*$ limit ({t2_limit_khz:.3f} kHz)",
    )
    axes[0].set_ylabel("RMS FWHM (kHz)")
    normalized_axis = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda linewidth_khz: linewidth_khz / t2_limit_khz,
            lambda t2_units: t2_units * t2_limit_khz,
        ),
    )
    normalized_axis.set_ylabel(r"RMS FWHM / $[1/(\pi T_2^*)]$")
    axes[0].set_title("Shaped-pulse RMS and constant-pulse reference")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(length, rmse, "o-", color="#7a3e9d", label="RMSE")
    axes[1].set_xlabel("Pulse length (µs)")
    axes[1].set_ylabel(r"Experiment–simulation RMSE / $[1/(\pi T_2^*)]$")
    axes[1].grid(alpha=0.25)
    relative_axis = axes[1].twinx()
    relative_axis.plot(length, relative, "s--", color="#247659", label="Relative RMSE")
    relative_axis.set_ylabel("Relative RMSE")
    handles_1, labels_1 = axes[1].get_legend_handles_labels()
    handles_2, labels_2 = relative_axis.get_legend_handles_labels()
    axes[1].legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper left")
    path = output_dir / "fwhm_rms_vs_pulse_length_good_snr.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_selection(
    grouped_rows: dict[float, list[dict[str, float]]],
    masks: dict[float, np.ndarray],
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(5, 2, figsize=(11, 14), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.reshape(-1)
    for axis, summary in zip(axes, summaries, strict=True):
        length = float(summary["pulse_length_us"])
        rows = grouped_rows[length]
        mask = masks[length]
        amplitude = np.asarray([row["amplitude_v"] for row in rows])
        measured = np.asarray([row["measured_fwhm_t2_units"] for row in rows])
        simulated = np.asarray([row["simulated_fwhm_t2_units"] for row in rows])
        finite_rejected = np.isfinite(measured) & ~mask
        axis.scatter(amplitude[finite_rejected], measured[finite_rejected], s=10, color="#b8b8b8", label="Rejected")
        axis.plot(amplitude, simulated, color="#d95f02", linewidth=1.2, label="Simulation")
        axis.scatter(amplitude[mask], measured[mask], s=14, color="#1b6ca8", zorder=3, label="Accepted")
        if np.any(mask):
            axis.axvspan(np.min(amplitude[mask]), np.max(amplitude[mask]), color="#1b6ca8", alpha=0.08)
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
        axis.set_title(
            f"{length:g} µs · {summary['selected_points']} points · "
            f"band threshold {summary['snr_threshold']:.1f}"
        )
        axis.set_xlabel("Peak amplitude (V)")
        axis.set_ylabel(r"FWHM / $[1/(\pi T_2^*)]$")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Adaptive good-SNR amplitude-band selection (accepted raw SNR ≥ 3)")
    path = output_dir / "fwhm_good_snr_selection_by_length.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir = campaign_dir / "fwhm_analysis"
    comparison_path = output_dir / "fwhm_experiment_vs_simulation.csv"
    comparisons = read_comparisons(comparison_path)
    grouped: dict[float, list[dict[str, float]]] = {}
    for row in comparisons:
        grouped.setdefault(row["pulse_length_us"], []).append(row)

    summaries: list[dict[str, Any]] = []
    masks: dict[float, np.ndarray] = {}
    detuning_span_hz = float(manifest["frequency_span_mhz"]) * 1e6
    for length in sorted(grouped):
        summary, mask = summarize_length(
            length,
            grouped[length],
            detuning_span_hz=detuning_span_hz,
            args=args,
        )
        summaries.append(summary)
        masks[length] = mask

    csv_path = output_dir / "fwhm_rms_by_length_good_snr.csv"
    write_summary_csv(csv_path, summaries)
    t2_limit_hz = 1.0 / (np.pi * float(manifest["device"]["t2_star_s"]))
    rms_path = plot_rms(summaries, output_dir, t2_limit_hz=t2_limit_hz)
    selection_path = plot_selection(grouped, masks, summaries, output_dir)
    method_path = output_dir / "fwhm_rms_good_snr_method.json"
    method_path.write_text(
        json.dumps(
            {
                "method": "adaptive_contiguous_snr_band",
                "snr_floor": args.snr_floor,
                "relative_threshold": args.relative_threshold,
                "smoothing_points": args.smoothing_points,
                "max_gap_points": args.max_gap_points,
                "max_width_fraction": args.max_width_fraction,
                "max_adjacent_width_ratio": args.max_adjacent_width_ratio,
                "detuning_span_hz": detuning_span_hz,
                "outputs": [str(csv_path), str(rms_path), str(selection_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in (csv_path, rms_path, selection_path, method_path):
        print(path)


if __name__ == "__main__":
    main()
