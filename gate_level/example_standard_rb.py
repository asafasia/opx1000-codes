"""Single-qubit randomized benchmarking with Qiskit Experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit_experiments.library import StandardRB

from gate_level.runner import Runner, result_counts_list

QUBIT = "q1"
SHOTS = 1000
LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_SAMPLES = 5
SEED = 12345
OUTPUT_FIGURE = Path(__file__).with_name("standard_rb.png")


def survival_probability(counts: dict[str, int], outcome: str = "0") -> float:
    shots = sum(counts.values())
    if shots == 0:
        return float("nan")
    return counts.get(outcome, 0) / shots


def fit_rb_decay(
    lengths: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, float]:
    """Fit p(m) = a * alpha**m + b and return (a, alpha, b)."""
    try:
        from scipy.optimize import curve_fit

        def model(depth, a, alpha, b):
            return a * np.power(alpha, depth) + b

        popt, _ = curve_fit(
            model,
            lengths,
            probabilities,
            p0=(0.5, 0.98, 0.5),
            bounds=([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        centered = np.clip(probabilities - 0.5, 1e-9, None)
        slope, intercept = np.polyfit(lengths, np.log(centered), 1)
        return float(np.exp(intercept)), float(np.exp(slope)), 0.5


def plot_rb(
    lengths: list[int],
    mean_survival: list[float],
    std_survival: list[float],
    fit_params: tuple[float, float, float],
):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(
        lengths,
        mean_survival,
        yerr=std_survival,
        fmt="o",
        color="tab:blue",
        capsize=3,
        label="Measured",
    )

    a, alpha, b = fit_params
    fit_x = np.linspace(min(lengths), max(lengths), 200)
    fit_y = a * np.power(alpha, fit_x) + b
    epc = (1.0 - alpha) / 2.0
    ax.plot(fit_x, fit_y, color="tab:red", label=f"alpha={alpha:.5f}, EPC={epc:.3g}")

    ax.set_xlabel("Number of Clifford gates")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Standard RB on {QUBIT}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=160)
    return fig


def main() -> None:
    experiment = StandardRB(
        physical_qubits=(0,),
        lengths=LENGTHS,
        num_samples=NUM_SAMPLES,
        seed=SEED,
    )
    circuits = experiment.circuits()

    runner = Runner(
        qubit=QUBIT,
        reset_type="active",
        status_updates=True,
    )
    transpiled = runner.transpile(circuits)
    result = runner.submit(transpiled, shots=SHOTS, do_transpile=False)
    counts_by_circuit = result_counts_list(result)

    survival_by_length: dict[int, list[float]] = defaultdict(list)
    for circuit, counts in zip(transpiled, counts_by_circuit):
        length = int(circuit.metadata["xval"])
        survival_by_length[length].append(survival_probability(dict(counts)))

    sorted_lengths = sorted(survival_by_length)
    mean_survival = [
        float(np.mean(survival_by_length[length])) for length in sorted_lengths
    ]
    std_survival = [
        (
            float(np.std(survival_by_length[length], ddof=1))
            if len(survival_by_length[length]) > 1
            else 0.0
        )
        for length in sorted_lengths
    ]
    fit_params = fit_rb_decay(np.array(sorted_lengths), np.array(mean_survival))
    plot_rb(sorted_lengths, mean_survival, std_survival, fit_params)

    a, alpha, b = fit_params
    print(f"RB fit: a={a:.6f}, alpha={alpha:.6f}, b={b:.6f}, EPC={(1-alpha)/2:.6g}")
    for length, mean, std in zip(sorted_lengths, mean_survival, std_survival):
        print(f"m={length:3d} survival={mean:.4f} std={std:.4f}")
    print(f"\nSaved plot: {OUTPUT_FIGURE}")
    plt.show()


if __name__ == "__main__":
    main()
