"""Plot fixed-amplitude echo-Lorentzian comparison traces."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    REPO_ROOT
    / "codex"
    / "individual_amplitude_logs"
    / "q1_fixed_amplitudes_20260701_084013"
    / "manifest.csv",
    REPO_ROOT
    / "codex"
    / "individual_amplitude_logs"
    / "q1_fixed_amplitudes_20260701_084315"
    / "manifest.csv",
]
OUTPUT_DIR = (
    REPO_ROOT
    / "codex"
    / "individual_amplitude_logs"
    / "q1_fixed_amplitudes_comparison_20260701"
)


def main() -> None:
    traces = load_traces()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "no-echo": plot_mode(
            traces,
            "no-echo",
            "Lorentzian, no echo",
            OUTPUT_DIR / "fixed_amplitude_lorentzian_no_echo.png",
        ),
        "echo": plot_mode(
            traces,
            "echo",
            "Echo Lorentzian",
            OUTPUT_DIR / "fixed_amplitude_echo_lorentzian.png",
        ),
    }
    for path in paths.values():
        print(path)


def load_traces() -> list[dict[str, object]]:
    rows = latest_successful_rows()
    traces = []
    for row in sorted(rows.values(), key=lambda item: (item["echo_mode"], float(item["rabi_mhz"]))):
        run_dir = saved_run_directory(Path(row["log_path"]))
        sweep = np.load(run_dir / "sweep.npz")
        results = np.load(run_dir / "results.npz")
        detuning_mhz = np.asarray(sweep["detuning"], dtype=float) / 1e6
        state = np.asarray(results["state"], dtype=float)
        if state.ndim != 3:
            raise ValueError(f"Expected state shape (qubit, detuning, amp), got {state.shape}")
        state_trace = np.nanmean(state[0], axis=1)
        traces.append(
            {
                "echo_mode": row["echo_mode"],
                "rabi_mhz": float(row["rabi_mhz"]),
                "detuning_mhz": detuning_mhz,
                "state": state_trace,
                "run_dir": run_dir,
            }
        )
    return traces


def latest_successful_rows() -> dict[tuple[str, float], dict[str, str]]:
    rows: dict[tuple[str, float], dict[str, str]] = {}
    for manifest in MANIFESTS:
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["status"] != "ok":
                    continue
                key = (row["echo_mode"], round(float(row["rabi_mhz"]), 6))
                rows[key] = row
    return rows


def saved_run_directory(log_path: Path) -> Path:
    text = log_path.read_text(encoding="utf-8")
    match = re.search(r"Raw calibration results saved to (.+)", text)
    if not match:
        raise ValueError(f"No saved calibration directory found in {log_path}")
    return Path(match.group(1).strip())


def plot_mode(
    traces: list[dict[str, object]],
    mode: str,
    title: str,
    output_path: Path,
) -> Path:
    selected = [trace for trace in traces if trace["echo_mode"] == mode]
    if not selected:
        raise ValueError(f"No traces found for mode {mode!r}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for trace in sorted(selected, key=lambda item: item["rabi_mhz"]):
        ax.plot(
            trace["detuning_mhz"],
            trace["state"],
            linewidth=1.6,
            label=f"{trace['rabi_mhz']:.2f} MHz",
        )

    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Excitation")
    ax.grid(alpha=0.25)
    ax.legend(title="Rabi amplitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
