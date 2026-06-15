"""Simulate a one-spin classical Ising update using QUA control flow."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow direct execution and `python -m ising_machine.one_spin_experiment`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qm import SimulationConfig
from qm.qua import *

from Projects.ising_machine.utils import (
    assign_spin_from_signal,
    calculate_flip_energy,
    calculate_single_spin_energy,
    zero_temperature_update,
)
from quam_config import create_machine


DEFAULT_ITERATIONS = 20
DEFAULT_THRESHOLD = 0.75
DEFAULT_FIELD = 0.25
DEFAULT_LOW_SIGNAL = 0.5
DEFAULT_HIGH_SIGNAL = 1.0
DEFAULT_SIMULATION_DURATION = 20_000
DEFAULT_RESULTS_PATH = REPOSITORY_ROOT / "data" / "ising_machine" / "one_spin_experiment.csv"


def required_simulation_duration(machine, iterations):
    """Estimate enough 4 ns clock cycles to complete every loop iteration."""
    readout_length_ns = machine.qubits["q9"].resonator.operations["readout"].length
    pulse_cycles_per_iteration = 2 * readout_length_ns // 4
    control_flow_margin_per_iteration = 100
    return iterations * (pulse_cycles_per_iteration + control_flow_margin_per_iteration) + 1_000


def build_one_spin_program(
    machine,
    iterations: int,
    threshold: float,
    field: float,
    low_signal: float,
    high_signal: float,
):
    """Build the controller-only one-spin Ising experiment."""
    output = machine.qubits["q9"].resonator

    with program() as one_spin_program:
        iteration = declare(int)
        signal = declare(fixed)
        qua_field = declare(fixed, value=field)
        initial_state = declare(int)
        final_state = declare(int)
        spin = declare(int)
        energy_before = declare(fixed)
        energy_after = declare(fixed)
        delta_energy = declare(fixed)
        flipped = declare(int)

        signal_stream = declare_stream()
        initial_state_stream = declare_stream()
        final_state_stream = declare_stream()
        spin_stream = declare_stream()
        energy_before_stream = declare_stream()
        energy_after_stream = declare_stream()
        delta_energy_stream = declare_stream()
        flipped_stream = declare_stream()

        with for_(iteration, 0, iteration < iterations, iteration + 1):
            # Synthetic analog input: alternate between state-0 and state-1 levels.
            with if_(iteration < iterations // 2):
                assign(signal, low_signal)
            with else_():
                assign(signal, high_signal)

            # Emit the synthetic input level before classifying it.
            output.measure("readout", amplitude_scale=signal)

            assign_spin_from_signal(signal, threshold, initial_state, spin)
            calculate_single_spin_energy(spin, qua_field, energy_before)
            calculate_flip_energy(spin, qua_field, delta_energy)
            assign(final_state, initial_state)
            zero_temperature_update(final_state, spin, delta_energy, flipped)
            calculate_single_spin_energy(spin, qua_field, energy_after)

            # Emit an analog pulse representing the updated classical state.
            with if_(final_state == 1):
                output.measure("readout", amplitude_scale=high_signal)
            with else_():
                output.measure("readout", amplitude_scale=low_signal)

            save(signal, signal_stream)
            save(initial_state, initial_state_stream)
            save(final_state, final_state_stream)
            save(spin, spin_stream)
            save(energy_before, energy_before_stream)
            save(energy_after, energy_after_stream)
            save(delta_energy, delta_energy_stream)
            save(flipped, flipped_stream)

        with stream_processing():
            signal_stream.save_all("signal")
            initial_state_stream.save_all("initial_state")
            final_state_stream.save_all("final_state")
            spin_stream.save_all("spin")
            energy_before_stream.save_all("energy_before")
            energy_after_stream.save_all("energy_after")
            delta_energy_stream.save_all("delta_energy")
            flipped_stream.save_all("flipped")

    return one_spin_program


def fetch_results(job):
    """Fetch all saved Ising variables from a completed job."""
    handles = job.result_handles
    handles.wait_for_all_values()
    names = (
        "signal",
        "initial_state",
        "final_state",
        "spin",
        "energy_before",
        "energy_after",
        "delta_energy",
        "flipped",
    )
    results = {}
    for name in names:
        values = np.asarray(handles.get(name).fetch_all())
        if values.dtype.names:
            values = values[values.dtype.names[0]]
        results[name] = np.asarray(values).reshape(-1)
    return results


def plot_results(results, threshold):
    """Plot the one-spin classification and zero-temperature update."""
    iteration = np.arange(len(results["signal"]))
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(iteration, results["signal"], "o-", label="synthetic signal")
    axes[0].axhline(threshold, color="black", linestyle="--", label="threshold")
    axes[0].set_ylabel("Signal")
    axes[0].legend()

    axes[1].step(iteration, results["initial_state"], where="mid", label="initial state")
    axes[1].step(iteration, results["final_state"], where="mid", label="updated state")
    axes[1].scatter(iteration, results["flipped"], marker="x", label="flipped")
    axes[1].set_ylabel("Binary value")
    axes[1].set_yticks([0, 1])
    axes[1].legend()

    axes[2].plot(iteration, results["energy_before"], "o-", label="energy before")
    axes[2].plot(iteration, results["energy_after"], "o-", label="energy after")
    axes[2].plot(iteration, results["delta_energy"], "o--", label="flip cost")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Energy")
    axes[2].legend()

    fig.suptitle("One-Spin Classical Ising Update")
    fig.tight_layout()


def plot_simulated_waveforms(job):
    """Plot the emitted synthetic-input and updated-state pulses."""
    samples = job.get_simulated_samples()
    controller_names = list(samples.keys())
    fig, axes = plt.subplots(
        len(controller_names),
        1,
        squeeze=False,
        figsize=(12, 4 * len(controller_names)),
    )
    for index, controller_name in enumerate(controller_names):
        plt.sca(axes[index, 0])
        samples[controller_name].plot()
        plt.title(f"{controller_name}: input pulse followed by updated-state pulse")
    fig.tight_layout()


def save_results_csv(results, path, threshold, field):
    """Save QUA experiment results in a format used by the reference model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "signal",
        "initial_state",
        "final_state",
        "spin",
        "energy_before",
        "energy_after",
        "delta_energy",
        "flipped",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("iteration", "threshold", "field", *columns))
        for index in range(len(results["signal"])):
            writer.writerow(
                (
                    index,
                    threshold,
                    field,
                    *(results[column][index] for column in columns),
                )
            )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument("--low-signal", type=float, default=DEFAULT_LOW_SIGNAL)
    parser.add_argument("--high-signal", type=float, default=DEFAULT_HIGH_SIGNAL)
    parser.add_argument(
        "--simulation-duration",
        type=int,
        default=DEFAULT_SIMULATION_DURATION,
        help="Simulation duration in 4 ns clock cycles.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="CSV path used by the pure-Python reference comparison.",
    )
    args = parser.parse_args()

    machine = create_machine()
    qua_program = build_one_spin_program(
        machine,
        args.iterations,
        args.threshold,
        args.field,
        args.low_signal,
        args.high_signal,
    )
    qmm = machine.connect()
    simulation_duration = max(
        args.simulation_duration,
        required_simulation_duration(machine, args.iterations),
    )
    print(f"Simulation duration: {simulation_duration} clock cycles")
    job = qmm.simulate(
        machine.generate_config(),
        qua_program,
        SimulationConfig(
            duration=simulation_duration,
            include_analog_waveforms=True,
        ),
    )
    job.wait_until("Done", timeout=120)
    results = fetch_results(job)
    results_path = save_results_csv(results, args.results_path, args.threshold, args.field)
    print(f"Saved results to {results_path}")

    completed_iterations = len(results["signal"])
    if completed_iterations != args.iterations:
        print(
            f"Warning: requested {args.iterations} iterations, "
            f"but simulation returned {completed_iterations}."
        )

    print("iteration signal initial final spin dE flipped energy")
    for index in range(completed_iterations):
        print(
            index,
            results["signal"][index],
            results["initial_state"][index],
            results["final_state"][index],
            results["spin"][index],
            results["delta_energy"][index],
            results["flipped"][index],
            results["energy_after"][index],
        )

    if not args.no_plot:
        plot_results(results, args.threshold)
        try:
            plot_simulated_waveforms(job)
        except Exception as exc:
            print(f"Could not plot simulated waveforms: {exc}")
        plt.show()


if __name__ == "__main__":
    main()
