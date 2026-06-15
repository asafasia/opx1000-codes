"""Pure-Python reference model for the one-spin QUA experiment."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = REPOSITORY_ROOT / "data" / "ising_machine" / "one_spin_experiment.csv"
DEFAULT_ITERATIONS = 20
DEFAULT_THRESHOLD = 0.75
DEFAULT_FIELD = 0.25
DEFAULT_LOW_SIGNAL = 0.5
DEFAULT_HIGH_SIGNAL = 1.0
INTEGER_COLUMNS = ("initial_state", "final_state", "spin", "flipped")
FLOAT_COLUMNS = ("signal", "energy_before", "energy_after", "delta_energy")


def simulate_signal(signal, threshold, field):
    """Return the expected one-spin Ising update for a single signal."""
    initial_state = 1 if signal > threshold else 0
    spin_before = -1 if initial_state == 1 else 1
    energy_before = -field * spin_before
    delta_energy = 2 * field * spin_before
    flipped = 1 if delta_energy < 0 else 0
    final_state = 1 - initial_state if flipped else initial_state
    spin_after = -spin_before if flipped else spin_before
    energy_after = -field * spin_after
    return {
        "initial_state": initial_state,
        "final_state": final_state,
        "spin": spin_after,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "delta_energy": delta_energy,
        "flipped": flipped,
    }


def load_experiment_csv(path):
    """Load experiment CSV columns as numeric NumPy arrays."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No experiment results found in {path}")

    data = {}
    for column in rows[0]:
        converter = int if column in ("iteration", *INTEGER_COLUMNS) else float
        data[column] = np.asarray([converter(row[column]) for row in rows])
    return data


def build_reference(experiment):
    """Calculate expected values using signals and parameters from the experiment."""
    reference = {"signal": experiment["signal"].copy()}
    expected_rows = [
        simulate_signal(signal, threshold, field)
        for signal, threshold, field in zip(
            experiment["signal"],
            experiment["threshold"],
            experiment["field"],
        )
    ]
    for column in (*INTEGER_COLUMNS, *FLOAT_COLUMNS[1:]):
        reference[column] = np.asarray([row[column] for row in expected_rows])
    return reference


def build_standalone_reference(iterations, threshold, field, low_signal, high_signal):
    """Generate the expected experiment data without requiring a QUA CSV."""
    split = iterations // 2
    signals = np.asarray(
        [low_signal if index < split else high_signal for index in range(iterations)]
    )
    experiment_shape = {
        "iteration": np.arange(iterations),
        "signal": signals,
        "threshold": np.full(iterations, threshold),
        "field": np.full(iterations, field),
    }
    reference = build_reference(experiment_shape)
    return {**experiment_shape, **reference}


def compare_results(experiment, reference):
    """Return a dictionary containing all columns that differ."""
    mismatches = {}
    for column in (*INTEGER_COLUMNS, *FLOAT_COLUMNS[1:]):
        actual = experiment[column]
        expected = reference[column]
        equal = np.array_equal(actual, expected) if column in INTEGER_COLUMNS else np.allclose(actual, expected)
        if not equal:
            mismatches[column] = {"actual": actual, "expected": expected}
    return mismatches


def plot_comparison(experiment, reference):
    """Plot experiment values beside pure-Python reference values."""
    iteration = experiment["iteration"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(iteration, experiment["signal"], "o-", label="experiment signal")
    axes[0].step(iteration, experiment["final_state"], where="mid", label="experiment final")
    axes[0].step(iteration, reference["final_state"], where="mid", linestyle="--", label="reference final")
    axes[0].legend()

    axes[1].plot(iteration, experiment["delta_energy"], "o-", label="experiment dE")
    axes[1].plot(iteration, reference["delta_energy"], "--", label="reference dE")
    axes[1].plot(iteration, experiment["flipped"], "x", label="experiment flipped")
    axes[1].legend()

    axes[2].plot(iteration, experiment["energy_after"], "o-", label="experiment energy")
    axes[2].plot(iteration, reference["energy_after"], "--", label="reference energy")
    axes[2].set_xlabel("Iteration")
    axes[2].legend()

    fig.suptitle("QUA Experiment vs Pure-Python Reference")
    fig.tight_layout()
    plt.show()


def plot_reference(reference):
    """Plot a standalone pure-Python reference simulation."""
    iteration = reference["iteration"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    axes[0].plot(iteration, reference["signal"], "o-", label="synthetic signal")
    axes[0].step(iteration, reference["final_state"], where="mid", label="final state")
    axes[0].legend()
    axes[1].plot(iteration, reference["delta_energy"], "o-", label="flip cost")
    axes[1].plot(iteration, reference["flipped"], "x", label="flipped")
    axes[1].legend()
    axes[2].plot(iteration, reference["energy_after"], "o-", label="final energy")
    axes[2].set_xlabel("Iteration")
    axes[2].legend()
    fig.suptitle("Pure-Python One-Spin Ising Reference")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument("--low-signal", type=float, default=DEFAULT_LOW_SIGNAL)
    parser.add_argument("--high-signal", type=float, default=DEFAULT_HIGH_SIGNAL)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.results_path.exists():
        reference = build_standalone_reference(
            args.iterations,
            args.threshold,
            args.field,
            args.low_signal,
            args.high_signal,
        )
        print(
            f"No experiment CSV found at {args.results_path}. "
            f"Generated standalone reference for {args.iterations} iterations."
        )
        if not args.no_plot:
            plot_reference(reference)
        return

    experiment = load_experiment_csv(args.results_path)
    reference = build_reference(experiment)
    mismatches = compare_results(experiment, reference)

    if mismatches:
        print("Comparison failed. Mismatched columns:")
        for column, values in mismatches.items():
            print(f"- {column}")
            print(f"  experiment: {values['actual']}")
            print(f"  reference:  {values['expected']}")
        raise SystemExit(1)

    print(f"Comparison passed for {len(experiment['iteration'])} iterations.")
    if not args.no_plot:
        plot_comparison(experiment, reference)


if __name__ == "__main__":
    main()
