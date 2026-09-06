"""Render the completed q6 beta=0 and beta=-0.22 spectroscopy maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "drag_beta_kappa_calibration"
    / "2026-09-05_20-37-12"
)
OUTPUT_DIR = CAMPAIGN_DIR / "figures"
RUNS = (
    (
        "beta_0",
        r"data\calibrations\2026-09-05\drag_kappa_joint_01\22-25-01-799230",
        "q6_beta_0_50us.png",
    ),
    (
        "drag_beta_-0.22",
        r"data\calibrations\2026-09-06\drag_kappa_joint_02\00-12-36-749691",
        "q6_beta_minus_0p22_50us.png",
    ),
)


def load_run(label: str, relative_path: str, records: dict[str, dict]):
    run_directory = REPOSITORY_ROOT / relative_path
    with np.load(run_directory / "sweep.npz", allow_pickle=False) as sweep:
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(run_directory / "results.npz", allow_pickle=False) as results:
        state = np.asarray(results["state"], dtype=float)[0].T
    metadata = json.loads((run_directory / "metadata.json").read_text())
    peak_amplitude_v = float(metadata["pulse"]["lorentzian_peak_amplitude"])
    amplitude_v = amp_prefactor * peak_amplitude_v
    rabi_mhz = np.asarray(records[label]["rabi_frequency_mhz"], dtype=float)
    return {
        "label": label,
        "detuning_mhz": detuning_mhz,
        "amplitude_v": amplitude_v,
        "rabi_mhz": rabi_mhz,
        "state": state,
        "metadata": metadata,
    }


def decorate_axis(axis, run: dict[str, object]) -> None:
    label = str(run["label"])
    beta = 0.0 if label == "beta_0" else -0.22
    axis.set_title(rf"q6: $\beta={beta:g}$, $\kappa=0$")
    axis.set_xlabel("Detuning [MHz]")
    axis.set_ylabel("Calibrated Rabi frequency [MHz]")
    rabi_per_volt = float(np.asarray(run["rabi_mhz"])[-1])
    secondary = axis.secondary_yaxis(
        "right",
        functions=(
            lambda rabi: rabi / rabi_per_volt,
            lambda voltage: voltage * rabi_per_volt,
        ),
    )
    secondary.set_ylabel("Pulse peak amplitude [V]")


def plot_individual(run: dict[str, object], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    image = axis.pcolormesh(
        run["detuning_mhz"],
        run["rabi_mhz"],
        run["state"],
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=0.6,
        rasterized=True,
    )
    decorate_axis(axis, run)
    colorbar = figure.colorbar(image, ax=axis, pad=0.12)
    colorbar.set_label("Measured excited-state probability")
    figure.suptitle("50 us echo root-Lorentzian spectroscopy, 2,000 averages")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_comparison(runs: list[dict[str, object]], output_path: Path) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, run in zip(axes, runs, strict=True):
        image = axis.pcolormesh(
            run["detuning_mhz"],
            run["rabi_mhz"],
            run["state"],
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=0.6,
            rasterized=True,
        )
        decorate_axis(axis, run)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, pad=0.08)
    colorbar.set_label("Measured excited-state probability")
    figure.suptitle(
        "q6 — 50 us echo root-Lorentzian spectroscopy, 2,000 averages, 0–1 V"
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_list = json.loads((CAMPAIGN_DIR / "records.json").read_text())
    records = {record["label"]: record for record in records_list}
    runs = [load_run(label, path, records) for label, path, _ in RUNS]
    for run, (_, _, filename) in zip(runs, RUNS, strict=True):
        plot_individual(run, OUTPUT_DIR / filename)
    comparison_path = OUTPUT_DIR / "q6_beta_comparison_50us.png"
    plot_comparison(runs, comparison_path)
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
