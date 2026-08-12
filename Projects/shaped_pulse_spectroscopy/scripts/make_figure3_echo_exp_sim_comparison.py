"""Compare the 2026-08-10 q1 data with matched three-level simulations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = ROOT.parent / "amplitude-robust-spectrscopy"
EXPERIMENT_ROOT = REFERENCE_ROOT / "data/experimental/2026-08-10/echo_lorentzian"
EXPERIMENT_ECHO_FALSE = EXPERIMENT_ROOT / "14-09-56-777281"
EXPERIMENT_ECHO_TRUE = EXPERIMENT_ROOT / "14-02-28-518579"
SIMULATION_ECHO_TRUE = ROOT / (
    "data/simulations/2026-08-10/echo_lorentzian/"
    "short_50x50_20us_c0.005_three_level_q1_t3000"
)
SIMULATION_ECHO_FALSE = ROOT / (
    "data/simulations/2026-08-10/echo_lorentzian/"
    "short_50x50_20us_c0.005_three_level_q1_no_echo_t3000"
)
OUTPUT = ROOT / (
    "data/analysis/2026-08-10/amplitude_robust_spectroscopy/"
    "figure3_left_echo_exp_sim"
)

TARGET_RABI_MHZ = np.asarray([2.5, 10.0, 25.0])


def nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(values - target))) for target in targets])


def load_experiment(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    parameters = json.loads((run_dir / "parameters.json").read_text())
    pulses = json.loads((run_dir / "profile/pulses.json").read_text())
    with np.load(run_dir / "sweep.npz", allow_pickle=False) as sweep:
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        amp_prefactor = np.asarray(sweep["amp_prefactor"], dtype=float)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        excited = np.asarray(results["state"], dtype=float)[0].T

    x180 = pulses["pulses"]["q1"]["x180_const"]
    pi_rabi_hz = 1.0 / (2.0 * float(x180["length_ns"]) * 1e-9)
    rabi_mhz = (
        amp_prefactor
        * float(parameters["lorentzian_peak_amplitude"])
        / float(x180["amplitude"])
        * pi_rabi_hz
        / 1e6
    )
    return detuning_mhz, rabi_mhz, excited, parameters


def load_simulation(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with xr.open_dataset(run_dir / "echo_lorentzian_qutip.nc") as dataset:
        detuning_mhz = np.asarray(dataset["detuning"].values, dtype=float) / 1e6
        rabi_mhz = np.asarray(
            dataset["rabi_frequency_hz"].sel(qubit="q1").values,
            dtype=float,
        ) / 1e6
        excited = np.asarray(dataset["state"].sel(qubit="q1").values, dtype=float).T
        attrs = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in dataset.attrs.items()
        }
    return detuning_mhz, rabi_mhz, excited, attrs


def write_trace_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    exp_noecho = load_experiment(EXPERIMENT_ECHO_FALSE)
    exp_echo = load_experiment(EXPERIMENT_ECHO_TRUE)
    sim_noecho = load_simulation(SIMULATION_ECHO_FALSE)
    sim_echo = load_simulation(SIMULATION_ECHO_TRUE)

    exp_noecho_indices = nearest_indices(exp_noecho[1], TARGET_RABI_MHZ)
    exp_echo_indices = nearest_indices(exp_echo[1], TARGET_RABI_MHZ)
    sim_noecho_indices = nearest_indices(sim_noecho[1], TARGET_RABI_MHZ)
    sim_echo_indices = nearest_indices(sim_echo[1], TARGET_RABI_MHZ)

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(6.8, 7.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_labels = iter("abcdef")
    trace_rows: list[dict[str, object]] = []

    for row, target in enumerate(TARGET_RABI_MHZ):
        exp_noecho_trace = exp_noecho[2][exp_noecho_indices[row]]
        exp_echo_trace = exp_echo[2][exp_echo_indices[row]]
        sim_noecho_trace = sim_noecho[2][sim_noecho_indices[row]]
        sim_echo_trace = sim_echo[2][sim_echo_indices[row]]

        left, right = axes[row]
        for ax, experiment, simulation, exp_trace, sim_trace in (
            (left, exp_noecho, sim_noecho, exp_noecho_trace, sim_noecho_trace),
            (right, exp_echo, sim_echo, exp_echo_trace, sim_echo_trace),
        ):
            ax.scatter(
                experiment[0],
                exp_trace,
                s=8,
                color="#1f77b4",
                alpha=0.68,
                linewidths=0,
                label="Experiment (10 Aug)",
                zorder=2,
            )
            ax.plot(
                simulation[0],
                sim_trace,
                color="#d95f02",
                lw=1.7,
                label="3-level simulation",
                zorder=3,
            )

        for column, ax in enumerate((left, right)):
            ax.axvline(0.0, color="0.65", lw=0.65, ls="--", zorder=0)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0.0, 0.82)
            ax.grid(alpha=0.15, lw=0.5)
            ax.text(
                0.035,
                0.95,
                f"({next(panel_labels)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
            if column == 0:
                ax.set_ylabel(r"Excited-state probability $P_e$")
                ax.text(
                    0.96,
                    0.94,
                    rf"target $\Omega_0/2\pi={target:g}$ MHz",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.8,
                )
            if row == 2:
                ax.set_xlabel(r"Drive detuning $(f_d-f_{01})$ (MHz)")

        if row == 0:
            left.set_title("Current root-Lorentzian\nexperiment + 3-level simulation, echo=False")
            right.set_title("Current echo-root-Lorentzian\nexperiment + 3-level simulation, echo=True")
            left.legend(loc="lower left", frameon=False)
            right.legend(loc="lower left", frameon=False)

        series = (
            ("current_noecho_experiment", exp_noecho[1][exp_noecho_indices[row]], exp_noecho[0], exp_noecho_trace),
            ("current_noecho_three_level_simulation", sim_noecho[1][sim_noecho_indices[row]], sim_noecho[0], sim_noecho_trace),
            ("current_echo_experiment", exp_echo[1][exp_echo_indices[row]], exp_echo[0], exp_echo_trace),
            ("current_echo_three_level_simulation", sim_echo[1][sim_echo_indices[row]], sim_echo[0], sim_echo_trace),
        )
        for name, actual_rabi, detuning, population in series:
            trace_rows.extend(
                {
                    "series": name,
                    "target_rabi_mhz": float(target),
                    "actual_rabi_mhz": float(actual_rabi),
                    "detuning_mhz": float(x),
                    "excited_probability": float(y),
                }
                for x, y in zip(detuning, population)
            )

    fig.suptitle(
        "Root-Lorentzian spectroscopy: current q1 experiment vs simulation",
        fontsize=9.4,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"figure3_left_echo_exp_sim.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_trace_rows(OUTPUT / "figure3_left_echo_exp_sim_traces.csv", trace_rows)
    provenance = {
        "description": "Replacement for the left side of amplitude-robust-spectroscopy Figure 3",
        "target_rabi_mhz": TARGET_RABI_MHZ.tolist(),
        "current_noecho_experiment": {
            "path": str(EXPERIMENT_ECHO_FALSE),
            "run_id": EXPERIMENT_ECHO_FALSE.name,
            "cutoff": float(exp_noecho[3]["cutoff"]),
            "duration_us": float(exp_noecho[3]["lorentzian_length_in_ns"]) / 1000.0,
            "shots_requested": int(exp_noecho[3]["num_shots"]),
            "actual_rabi_mhz": exp_noecho[1][exp_noecho_indices].tolist(),
        },
        "current_noecho_simulation": {
            "path": str(SIMULATION_ECHO_FALSE / "echo_lorentzian_qutip.nc"),
            "model": "three-level Duffing transmon",
            "actual_rabi_mhz": sim_noecho[1][sim_noecho_indices].tolist(),
            "parameters": sim_noecho[3],
        },
        "current_echo_experiment": {
            "path": str(EXPERIMENT_ECHO_TRUE),
            "run_id": EXPERIMENT_ECHO_TRUE.name,
            "cutoff": float(exp_echo[3]["cutoff"]),
            "duration_us": float(exp_echo[3]["lorentzian_length_in_ns"]) / 1000.0,
            "shots_requested": int(exp_echo[3]["num_shots"]),
            "actual_rabi_mhz": exp_echo[1][exp_echo_indices].tolist(),
        },
        "current_echo_simulation": {
            "path": str(SIMULATION_ECHO_TRUE / "echo_lorentzian_qutip.nc"),
            "model": "three-level Duffing transmon",
            "actual_rabi_mhz": sim_echo[1][sim_echo_indices].tolist(),
            "parameters": sim_echo[3],
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
