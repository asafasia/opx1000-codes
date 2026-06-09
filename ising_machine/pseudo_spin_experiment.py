"""Run a probabilistic Ising machine using readout-pulse amplitudes as spins."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.results import fetching_tool, progress_counter

# Allow direct execution and `python -m ising_machine.pseudo_spin_experiment`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qm.qua import *

from ising_machine.model import (
    IsingProblem,
    IsingRun,
    SPIN_DOWN_AMPLITUDE,
    SPIN_UP_AMPLITUDE,
    ring_couplings,
)


DEFAULT_N_SPINS = 5000
DEFAULT_ITERATIONS = 2000
DEFAULT_COUPLING = 0.25
DEFAULT_FIELD = 0.0
DEFAULT_TEMPERATURE = 0.13
DEFAULT_SEED = 7
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "ising_machine" / "pseudo_spin_run.csv"


def uniform_ring_parameters(problem: IsingProblem) -> tuple[float, float] | None:
    """Return uniform ring coupling and field, or None for a general problem."""
    if problem.n_spins < 3 or not np.allclose(problem.fields, problem.fields[0]):
        return None
    coupling = float(problem.couplings[0, 1])
    if np.allclose(problem.couplings, ring_couplings(problem.n_spins, coupling)):
        return coupling, float(problem.fields[0])
    return None


def validate_controller_ranges(
    problem: IsingProblem, temperature_start: float, temperature_end: float
) -> None:
    """Reject problems that exceed the QUA fixed-point and exp input ranges."""
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("temperatures must be positive")
    if temperature_start >= 8 or temperature_end >= 8:
        raise ValueError("temperatures must be below the QUA fixed-point limit of 8")
    max_local_field = np.max(
        np.abs(problem.fields) + np.sum(np.abs(problem.couplings), axis=1)
    )
    max_delta = 2.0 * max_local_field
    if max_local_field >= 4.0 or max_delta >= 8.0:
        raise ValueError("problem exceeds the QUA fixed-point range; reduce J or h")
    if max_delta / min(temperature_start, temperature_end) >= 8.0:
        raise ValueError("Boltzmann exponent exceeds QUA range; raise T or reduce J/h")


def build_pseudo_spin_program(
    machine,
    problem: IsingProblem,
    iterations: int,
    temperature_start: float,
    temperature_end: float,
    initial_spins: np.ndarray,
    random_seed: int,
    save_configurations: bool = True,
):
    """Build the controller-side pulse emission and Metropolis update loop."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    initial_spins = problem.validate_spins(initial_spins)
    validate_controller_ranges(problem, temperature_start, temperature_end)
    output = machine.qubits["q1"].resonator
    n_spins = problem.n_spins
    ring_parameters = uniform_ring_parameters(problem)
    temperature_step = (
        0.0
        if iterations == 1
        else (temperature_end - temperature_start) / (iterations - 1)
    )

    with program() as pseudo_spin_program:
        iteration = declare(int)
        spin_index = declare(int)
        attempt = declare(int)
        neighbor = declare(int)
        left_neighbor = declare(int)
        right_neighbor = declare(int)
        neighbor_sum = declare(int)
        spins = declare(int, value=initial_spins.tolist())
        if ring_parameters is None:
            couplings = declare(fixed, value=problem.couplings.reshape(-1).tolist())
            fields = declare(fixed, value=problem.fields.tolist())
        pulse_amplitude = declare(fixed)
        temperature = declare(fixed)
        local_field = declare(fixed)
        delta_energy = declare(fixed)
        acceptance_probability = declare(fixed)
        accepted = declare(int)
        accepted_count = declare(int)
        magnetization = declare(int)
        rng = Random(random_seed)

        if save_configurations:
            spin_stream = declare_stream()
        temperature_stream = declare_stream()
        accepted_count_stream = declare_stream()
        magnetization_stream = declare_stream()

        with for_(iteration, 0, iteration < iterations, iteration + 1):
            assign(
                temperature,
                temperature_start + Cast.mul_fixed_by_int(temperature_step, iteration),
            )

            # Emit one pulse per pseudo-spin. No physical qubit state is involved.
            with for_(spin_index, 0, spin_index < n_spins, spin_index + 1):
                with if_(spins[spin_index] == 1):
                    assign(pulse_amplitude, SPIN_UP_AMPLITUDE)
                with else_():
                    assign(pulse_amplitude, SPIN_DOWN_AMPLITUDE)
                output.measure("readout", amplitude_scale=pulse_amplitude)

            # Random-site updates avoid directional drift from an ordered sweep.
            assign(accepted_count, 0)
            with for_(attempt, 0, attempt < n_spins, attempt + 1):
                assign(spin_index, rng.rand_int(n_spins))
                if ring_parameters is not None:
                    ring_coupling, ring_field = ring_parameters
                    assign(left_neighbor, spin_index - 1)
                    with if_(spin_index == 0):
                        assign(left_neighbor, n_spins - 1)
                    assign(right_neighbor, spin_index + 1)
                    with if_(spin_index == n_spins - 1):
                        assign(right_neighbor, 0)
                    assign(
                        neighbor_sum,
                        spins[left_neighbor] + spins[right_neighbor],
                    )
                    assign(
                        local_field,
                        ring_field
                        + Cast.mul_fixed_by_int(ring_coupling, neighbor_sum),
                    )
                else:
                    assign(local_field, fields[spin_index])
                    with for_(neighbor, 0, neighbor < n_spins, neighbor + 1):
                        assign(
                            local_field,
                            local_field
                            + Cast.mul_fixed_by_int(
                                couplings[spin_index * n_spins + neighbor],
                                spins[neighbor],
                            ),
                        )
                assign(
                    delta_energy,
                    2.0 * Cast.mul_fixed_by_int(local_field, spins[spin_index]),
                )
                assign(accepted, 0)
                with if_(delta_energy <= 0):
                    assign(accepted, 1)
                with else_():
                    assign(
                        acceptance_probability,
                        Math.exp(-delta_energy / temperature),
                    )
                    with if_(rng.rand_fixed() < acceptance_probability):
                        assign(accepted, 1)
                with if_(accepted == 1):
                    assign(spins[spin_index], -spins[spin_index])
                    assign(accepted_count, accepted_count + 1)

            # Always calculate magnetization; configurations are optional.
            assign(magnetization, 0)
            with for_(spin_index, 0, spin_index < n_spins, spin_index + 1):
                assign(magnetization, magnetization + spins[spin_index])
                if save_configurations:
                    save(spins[spin_index], spin_stream)
            save(temperature, temperature_stream)
            save(accepted_count, accepted_count_stream)
            save(magnetization, magnetization_stream)

        with stream_processing():
            if save_configurations:
                spin_stream.buffer(n_spins).save_all("configurations")
            temperature_stream.save_all("temperatures")
            accepted_count_stream.save_all("accepted_flips")
            magnetization_stream.save_all("magnetizations")

    return pseudo_spin_program


def _fetch_array(job, name: str) -> np.ndarray:
    values = np.asarray(job.result_handles.get(name).fetch_all())
    return _normalize_fetched_values(values)


def _normalize_fetched_values(values) -> np.ndarray:
    """Convert live or completed result-handle values into a plain array."""
    values = np.asarray(values)
    if values.dtype.names:
        values = values[values.dtype.names[0]]
    return np.asarray(values)


def fetch_run(job, problem: IsingProblem, full_analysis: bool = False) -> IsingRun:
    """Fetch saved controller state and calculate host-side observables."""
    job.result_handles.wait_for_all_values()
    temperatures = _fetch_array(job, "temperatures").reshape(-1)
    if not full_analysis:
        magnetizations = _fetch_array(job, "magnetizations").reshape(-1)
        return IsingRun.from_lightweight_metrics(
            temperatures,
            magnetizations,
            problem.n_spins,
        )

    configurations = _fetch_array(job, "configurations").reshape(-1, problem.n_spins)
    accepted_flips = _fetch_array(job, "accepted_flips").reshape(-1)
    return IsingRun.from_configurations(
        problem,
        configurations,
        temperatures,
        accepted_flips,
        full_analysis=full_analysis,
    )


def fetch_run_live(
    job,
    problem: IsingProblem,
    iterations: int,
    refresh_interval: float,
    full_analysis: bool = False,
) -> IsingRun:
    """Fetch completed iterations and update a live spin-evolution heatmap."""
    results = fetching_tool(
        job,
        data_list=["configurations", "temperatures", "accepted_flips"],
        mode="live",
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    image = axis.imshow(
        np.zeros((problem.n_spins, 1)),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Pseudo-spin")
    figure.colorbar(image, ax=axis, label="Spin")
    figure.tight_layout()
    plt.show(block=False)

    latest = None
    completed = 0
    while results.is_processing():
        fetched = results.fetch_all()
        configurations = _normalize_fetched_values(fetched[0])
        temperatures = _normalize_fetched_values(fetched[1]).reshape(-1)
        accepted_flips = _normalize_fetched_values(fetched[2]).reshape(-1)
        if configurations.size == 0:
            plt.pause(refresh_interval)
            continue

        configurations = configurations.reshape(-1, problem.n_spins)
        common_length = min(
            len(configurations),
            len(temperatures),
            len(accepted_flips),
        )
        if common_length == 0:
            plt.pause(refresh_interval)
            continue

        latest = (
            configurations[:common_length],
            temperatures[:common_length],
            accepted_flips[:common_length],
        )
        if common_length != completed:
            completed = common_length
            image.set_data(latest[0].T)
            image.set_extent((-0.5, completed - 0.5, -0.5, problem.n_spins - 0.5))
            axis.set_xlim(-0.5, max(0.5, completed - 0.5))
            axis.set_title(f"Live Spin Evolution: {completed}/{iterations} iterations")
            progress_counter(
                completed,
                iterations,
                start_time=results.get_start_time(),
            )
        plt.pause(refresh_interval)

    if latest is None:
        raise RuntimeError("The live job completed without returning configurations.")
    return IsingRun.from_configurations(
        problem,
        *latest,
        full_analysis=full_analysis,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-spins", type=int, default=DEFAULT_N_SPINS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--final-temperature",
        type=float,
        help="Optional linear annealing endpoint.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Update the spin-evolution heatmap while the hardware job runs.",
    )
    parser.add_argument(
        "--live-refresh",
        type=float,
        default=0.2,
        help="Seconds between live-plot refreshes.",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Calculate energy, heat capacity, susceptibility, and correlation length.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    from quam_config import create_machine

    args = parse_args()
    if args.n_spins < 2:
        raise ValueError("n-spins must be at least 2")
    if args.live_refresh <= 0:
        raise ValueError("live-refresh must be positive")
    final_temperature = (
        args.temperature if args.final_temperature is None else args.final_temperature
    )
    problem = IsingProblem(
        ring_couplings(args.n_spins, args.coupling),
        np.full(args.n_spins, args.field),
    )
    initial_spins = np.random.default_rng(args.seed).choice(
        (-1, 1), size=args.n_spins
    )
    machine = create_machine()
    qua_program = build_pseudo_spin_program(
        machine,
        problem,
        args.iterations,
        args.temperature,
        final_temperature,
        initial_spins,
        args.seed,
        save_configurations=args.live_plot or args.full_analysis,
    )
    qmm = machine.connect()
    config = machine.generate_config()
    qm = qmm.open_qm(config)
    run_started = time.perf_counter()
    job = qm.execute(qua_program)
    print("Pseudo-spin Ising experiment submitted to real OPX hardware.")

    run = (
        fetch_run_live(
            job,
            problem,
            args.iterations,
            args.live_refresh,
            full_analysis=args.full_analysis,
        )
        if args.live_plot
        else fetch_run(job, problem, full_analysis=args.full_analysis)
    )
    run_duration = time.perf_counter() - run_started
    output = run.save_csv(args.output, problem)
    print(f"Saved {len(run.configurations)} iterations to {output}")
    print(
        f"Run time={run_duration:.3f} s, "
        f"average={run_duration / len(run.configurations):.6f} s/iteration"
    )
    print(f"Final magnetization per spin={run.magnetizations[-1]:.6g}")
    if run.energies is not None:
        print(f"Final energy per spin={run.energies[-1]:.6g}")
    if not args.no_plot:
        run.plot()
        plt.show()


if __name__ == "__main__":
    main()
