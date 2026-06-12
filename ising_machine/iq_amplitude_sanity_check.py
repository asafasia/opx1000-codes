"""Acquire and plot IQ responses for readout amplitude scales 1.0 and 0.5."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow direct execution and `python -m ising_machine.iq_amplitude_sanity_check`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qm.qua import *
from qualang_tools.units import unit


DEFAULT_QUBIT = "q9"
DEFAULT_NUM_SHOTS = 1_0000
HIGH_AMPLITUDE = 1.0
LOW_AMPLITUDE = 0.5
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "ising_machine" / "iq_amplitude_check.csv"


def build_iq_amplitude_program(machine, qubit_name: str, num_shots: int):
    """Build paired IQ acquisitions at amplitude scales 1.0 and 0.5."""
    if num_shots < 1:
        raise ValueError("num_shots must be positive")

    u = unit(coerce_to_integer=True)
    resonator = machine.qubits[qubit_name].resonator

    with program() as iq_program:
        shot = declare(int)
        high_i = declare(fixed)
        high_q = declare(fixed)
        low_i = declare(fixed)
        low_q = declare(fixed)

        high_i_stream = declare_stream()
        high_q_stream = declare_stream()
        low_i_stream = declare_stream()
        low_q_stream = declare_stream()

        with for_(shot, 0, shot < num_shots, shot + 1):
            # Alternate acquisition order to avoid bias from slow hardware drift.
            with if_((shot >> 1) << 1 == shot):
                resonator.measure(
                    "readout",
                    amplitude_scale=HIGH_AMPLITUDE,
                    qua_vars=(high_i, high_q),
                )
                resonator.wait(machine.depletion_time * u.ns)
                resonator.measure(
                    "readout",
                    amplitude_scale=LOW_AMPLITUDE,
                    qua_vars=(low_i, low_q),
                )
            with else_():
                resonator.measure(
                    "readout",
                    amplitude_scale=LOW_AMPLITUDE,
                    qua_vars=(low_i, low_q),
                )
                resonator.wait(machine.depletion_time * u.ns)
                resonator.measure(
                    "readout",
                    amplitude_scale=HIGH_AMPLITUDE,
                    qua_vars=(high_i, high_q),
                )

            save(high_i, high_i_stream)
            save(high_q, high_q_stream)
            save(low_i, low_i_stream)
            save(low_q, low_q_stream)
            resonator.wait(machine.depletion_time * u.ns)

        with stream_processing():
            high_i_stream.save_all("high_i")
            high_q_stream.save_all("high_q")
            low_i_stream.save_all("low_i")
            low_q_stream.save_all("low_q")

    return iq_program


def _fetch_array(job, name: str) -> np.ndarray:
    values = np.asarray(job.result_handles.get(name).fetch_all())
    if values.dtype.names:
        values = values[values.dtype.names[0]]
    return np.asarray(values, dtype=float).reshape(-1)


def fetch_iq_results(job) -> dict[str, np.ndarray]:
    """Wait for the hardware job and fetch all paired IQ measurements."""
    job.result_handles.wait_for_all_values()
    return {
        name: _fetch_array(job, name)
        for name in ("high_i", "high_q", "low_i", "low_q")
    }


def save_results_csv(results: dict[str, np.ndarray], path: Path) -> Path:
    """Save paired raw IQ measurements."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("shot", "amplitude_scale", "I", "Q"))
        for shot in range(len(results["high_i"])):
            writer.writerow(
                (
                    shot,
                    HIGH_AMPLITUDE,
                    results["high_i"][shot],
                    results["high_q"][shot],
                )
            )
            writer.writerow(
                (
                    shot,
                    LOW_AMPLITUDE,
                    results["low_i"][shot],
                    results["low_q"][shot],
                )
            )
    return path


def iq_summary(results: dict[str, np.ndarray]) -> dict[str, float]:
    """Calculate centroid magnitudes and their ratio."""
    high_centroid = np.asarray([np.mean(results["high_i"]), np.mean(results["high_q"])])
    low_centroid = np.asarray([np.mean(results["low_i"]), np.mean(results["low_q"])])
    high_magnitude = float(np.linalg.norm(high_centroid))
    low_magnitude = float(np.linalg.norm(low_centroid))
    return {
        "high_i": float(high_centroid[0]),
        "high_q": float(high_centroid[1]),
        "low_i": float(low_centroid[0]),
        "low_q": float(low_centroid[1]),
        "high_magnitude": high_magnitude,
        "low_magnitude": low_magnitude,
        "magnitude_ratio": low_magnitude / high_magnitude if high_magnitude else np.nan,
    }


def plot_iq_results(results: dict[str, np.ndarray]):
    """Plot raw IQ clouds and mark their centroids."""
    summary = iq_summary(results)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        results["high_i"],
        results["high_q"],
        s=12,
        alpha=0.4,
        label="Amplitude 1.0",
    )
    axis.scatter(
        results["low_i"],
        results["low_q"],
        s=12,
        alpha=0.4,
        label="Amplitude 0.5",
    )
    axis.scatter(
        [summary["high_i"], summary["low_i"]],
        [summary["high_q"], summary["low_q"]],
        marker="x",
        s=140,
        linewidths=3,
        color="black",
        label="Centroids",
    )
    axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    axis.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    axis.set_xlabel("I")
    axis.set_ylabel("Q")
    axis.set_title("Readout IQ Response by Pulse Amplitude")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default=DEFAULT_QUBIT)
    parser.add_argument("--num-shots", type=int, default=DEFAULT_NUM_SHOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    from quam_config import create_machine

    args = parse_args()
    machine = create_machine()
    qua_program = build_iq_amplitude_program(machine, args.qubit, args.num_shots)
    qmm = machine.connect()
    qm = qmm.open_qm(machine.generate_config())

    started = time.perf_counter()
    job = qm.execute(qua_program)
    print("IQ amplitude sanity check submitted to real OPX hardware.")
    results = fetch_iq_results(job)
    duration = time.perf_counter() - started

    output = save_results_csv(results, args.output)
    summary = iq_summary(results)
    print(f"Saved {len(results['high_i'])} paired shots to {output}")
    print(f"Run time={duration:.3f} s")
    print(
        f"Amplitude 1.0 centroid=({summary['high_i']:.6g}, {summary['high_q']:.6g}), "
        f"magnitude={summary['high_magnitude']:.6g}"
    )
    print(
        f"Amplitude 0.5 centroid=({summary['low_i']:.6g}, {summary['low_q']:.6g}), "
        f"magnitude={summary['low_magnitude']:.6g}"
    )
    print(f"Centroid magnitude ratio 0.5/1.0={summary['magnitude_ratio']:.6g}")

    if not args.no_plot:
        plot_iq_results(results)
        plt.show()


if __name__ == "__main__":
    main()
