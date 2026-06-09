"""Run an amplitude-encoded 2D nearest-neighbor Ising model on real OPX hardware."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.results import fetching_tool, progress_counter

# Allow direct execution and `python -m ising_machine.pseudo_spin_2d_experiment`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qm.qua import *

from ising_machine.model import SPIN_DOWN_AMPLITUDE, SPIN_UP_AMPLITUDE


DEFAULT_ROWS = 50
DEFAULT_COLS = 50
DEFAULT_ITERATIONS = 2000
DEFAULT_COUPLING = 0.2
DEFAULT_FIELD = 0.0
DEFAULT_TEMPERATURE = 0.42
DEFAULT_ITERATION_WAIT_MS = 50.0
DEFAULT_SEED = 7
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "ising_machine" / "pseudo_spin_2d_run.csv"


@dataclass(frozen=True)
class GridIsingProblem:
    """Uniform 2D nearest-neighbor Ising model with periodic boundaries."""

    rows: int
    cols: int
    coupling: float = DEFAULT_COUPLING
    field: float = DEFAULT_FIELD

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("rows and cols must both be at least 2")

    @property
    def n_spins(self) -> int:
        return self.rows * self.cols

    def validate_spins(self, spins: np.ndarray) -> np.ndarray:
        spins = np.asarray(spins, dtype=np.int8).reshape(-1)
        if spins.shape != (self.n_spins,) or not np.all(np.isin(spins, (-1, 1))):
            raise ValueError(f"spins must contain {self.n_spins} values of -1/+1")
        return spins

    def neighbor_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened periodic left, right, up, and down indices."""
        indices = np.arange(self.n_spins).reshape(self.rows, self.cols)
        return tuple(
            np.asarray(neighbors, dtype=int).reshape(-1)
            for neighbors in (
                np.roll(indices, 1, axis=1),
                np.roll(indices, -1, axis=1),
                np.roll(indices, 1, axis=0),
                np.roll(indices, -1, axis=0),
            )
        )

    def delta_energy(self, spins: np.ndarray, spin_index: int) -> float:
        spins = self.validate_spins(spins)
        neighbors = self.neighbor_indices()
        neighbor_sum = sum(spins[indices[spin_index]] for indices in neighbors)
        return float(
            2.0 * spins[spin_index] * (self.coupling * neighbor_sum + self.field)
        )

    def energy_per_spin(self, spins: np.ndarray) -> float:
        grid = self.validate_spins(spins).reshape(self.rows, self.cols)
        bonds = grid * (np.roll(grid, -1, axis=0) + np.roll(grid, -1, axis=1))
        energy = -self.coupling * np.sum(bonds) - self.field * np.sum(grid)
        return float(energy / self.n_spins)


def validate_controller_ranges(
    problem: GridIsingProblem,
    temperature_start: float,
    temperature_end: float,
) -> None:
    """Validate QUA fixed-point and exponential input ranges."""
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("temperatures must be positive")
    max_local_field = abs(problem.field) + 4 * abs(problem.coupling)
    max_delta = 2 * max_local_field
    if max_local_field >= 4 or max_delta >= 8:
        raise ValueError("problem exceeds QUA fixed-point range; reduce J or h")
    if max_delta / min(temperature_start, temperature_end) >= 8:
        raise ValueError("Boltzmann exponent exceeds QUA range; raise T or reduce J/h")


def build_2d_pseudo_spin_program(
    machine,
    problem: GridIsingProblem,
    iterations: int,
    temperature_start: float,
    temperature_end: float,
    initial_spins: np.ndarray,
    random_seed: int,
    save_configurations: bool = False,
    iteration_wait_ns: int = 0,
):
    """Build a random-site 2D Metropolis experiment using four grid neighbors."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if iteration_wait_ns < 0:
        raise ValueError("iteration_wait_ns cannot be negative")
    initial_spins = problem.validate_spins(initial_spins)
    validate_controller_ranges(problem, temperature_start, temperature_end)
    left, right, up, down = problem.neighbor_indices()
    output = machine.qubits["q1"].resonator
    n_spins = problem.n_spins
    temperature_step = (
        0.0
        if iterations == 1
        else (temperature_end - temperature_start) / (iterations - 1)
    )

    with program() as grid_program:
        iteration = declare(int)
        spin_index = declare(int)
        attempt = declare(int)
        neighbor_sum = declare(int)
        magnetization = declare(int)
        spins = declare(int, value=initial_spins.tolist())
        left_indices = declare(int, value=left.tolist())
        right_indices = declare(int, value=right.tolist())
        up_indices = declare(int, value=up.tolist())
        down_indices = declare(int, value=down.tolist())
        pulse_amplitude = declare(fixed)
        temperature = declare(fixed)
        local_field = declare(fixed)
        delta_energy = declare(fixed)
        acceptance_probability = declare(fixed)
        accepted = declare(int)
        rng = Random(random_seed)

        if save_configurations:
            spin_stream = declare_stream()
        temperature_stream = declare_stream()
        magnetization_stream = declare_stream()

        with for_(iteration, 0, iteration < iterations, iteration + 1):
            assign(
                temperature,
                temperature_start + Cast.mul_fixed_by_int(temperature_step, iteration),
            )

            with for_(spin_index, 0, spin_index < n_spins, spin_index + 1):
                with if_(spins[spin_index] == 1):
                    assign(pulse_amplitude, SPIN_UP_AMPLITUDE)
                with else_():
                    assign(pulse_amplitude, SPIN_DOWN_AMPLITUDE)
                output.measure("readout", amplitude_scale=pulse_amplitude)

            with for_(attempt, 0, attempt < n_spins, attempt + 1):
                assign(spin_index, rng.rand_int(n_spins))
                assign(
                    neighbor_sum,
                    spins[left_indices[spin_index]]
                    + spins[right_indices[spin_index]]
                    + spins[up_indices[spin_index]]
                    + spins[down_indices[spin_index]],
                )
                assign(
                    local_field,
                    problem.field
                    + Cast.mul_fixed_by_int(problem.coupling, neighbor_sum),
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

            assign(magnetization, 0)
            with for_(spin_index, 0, spin_index < n_spins, spin_index + 1):
                assign(magnetization, magnetization + spins[spin_index])
                if save_configurations:
                    save(spins[spin_index], spin_stream)
            save(temperature, temperature_stream)
            save(magnetization, magnetization_stream)
            if iteration_wait_ns:
                output.wait(iteration_wait_ns)

        with stream_processing():
            if save_configurations:
                spin_stream.buffer(n_spins).save_all("configurations")
            temperature_stream.save_all("temperatures")
            magnetization_stream.save_all("magnetizations")

    return grid_program


def _fetch_array(job, name: str) -> np.ndarray:
    values = np.asarray(job.result_handles.get(name).fetch_all())
    return _normalize_fetched_values(values)


def _normalize_fetched_values(values) -> np.ndarray:
    """Convert live or completed result-handle values into a plain array."""
    values = np.asarray(values)
    if values.dtype.names:
        values = values[values.dtype.names[0]]
    return np.asarray(values)


def fetch_2d_results_live(
    job,
    problem: GridIsingProblem,
    iterations: int,
    refresh_interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch completed grids and update a live 2D spin image."""
    results = fetching_tool(
        job,
        data_list=["configurations", "temperatures", "magnetizations"],
        mode="live",
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        np.zeros((problem.rows, problem.cols)),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    figure.colorbar(image, ax=axis, label="Spin")
    figure.tight_layout()
    plt.show(block=False)

    latest = None
    completed = 0
    while results.is_processing():
        fetched = results.fetch_all()
        configurations = _normalize_fetched_values(fetched[0])
        temperatures = _normalize_fetched_values(fetched[1]).reshape(-1)
        magnetizations = _normalize_fetched_values(fetched[2]).reshape(-1)
        if configurations.size == 0:
            plt.pause(refresh_interval)
            continue

        configurations = configurations.reshape(-1, problem.n_spins)
        common_length = min(
            len(configurations),
            len(temperatures),
            len(magnetizations),
        )
        if common_length == 0:
            plt.pause(refresh_interval)
            continue

        latest = (
            configurations[:common_length],
            temperatures[:common_length],
            magnetizations[:common_length],
        )
        if common_length != completed:
            completed = common_length
            image.set_data(latest[0][-1].reshape(problem.rows, problem.cols))
            axis.set_title(f"Live 2D Spin Grid: {completed}/{iterations} iterations")
            progress_counter(
                completed,
                iterations,
                start_time=results.get_start_time(),
            )
        plt.pause(refresh_interval)

    if latest is None:
        raise RuntimeError("The live job completed without returning configurations.")
    return latest


def save_metrics_csv(
    path: Path,
    temperatures: np.ndarray,
    magnetizations: np.ndarray,
) -> Path:
    """Save lightweight 2D run metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("iteration", "temperature", "magnetization_per_spin"))
        for iteration, (temperature, magnetization) in enumerate(
            zip(temperatures, magnetizations)
        ):
            writer.writerow((iteration, temperature, magnetization))
    return path


def plot_2d_results(
    problem: GridIsingProblem,
    temperatures: np.ndarray,
    magnetizations: np.ndarray,
    configurations: np.ndarray | None = None,
):
    """Plot final 2D domains when available and lightweight run metrics."""
    if configurations is None:
        fig, metric_axis = plt.subplots(figsize=(10, 5))
    else:
        fig, (grid_axis, metric_axis) = plt.subplots(1, 2, figsize=(13, 5))
        image = grid_axis.imshow(
            configurations[-1].reshape(problem.rows, problem.cols),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        grid_axis.set_title("Final 2D Spin Configuration")
        grid_axis.set_xlabel("Column")
        grid_axis.set_ylabel("Row")
        fig.colorbar(image, ax=grid_axis, label="Spin")

    iterations = np.arange(len(temperatures))
    temperature_axis = metric_axis.twinx()
    magnetization_line = metric_axis.plot(
        iterations,
        magnetizations,
        label="magnetization per spin",
    )[0]
    temperature_line = temperature_axis.plot(
        iterations,
        temperatures,
        label="temperature",
        color="tab:red",
    )[0]
    metric_axis.set_xlabel("Iteration")
    metric_axis.set_ylabel("Magnetization per spin")
    temperature_axis.set_ylabel("Temperature", color="tab:red")
    metric_axis.grid(alpha=0.25)
    metric_axis.legend(handles=(magnetization_line, temperature_line))
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument("--field", type=float, default=DEFAULT_FIELD)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--final-temperature", type=float)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-configurations", action="store_true")
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Update the 2D spin grid while the hardware job runs.",
    )
    parser.add_argument(
        "--live-refresh",
        type=float,
        default=0.2,
        help="Seconds between live-grid refreshes.",
    )
    parser.add_argument(
        "--iteration-wait-ms",
        type=float,
        default=DEFAULT_ITERATION_WAIT_MS,
        help="Controller-side delay between iterations in milliseconds.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live_refresh <= 0:
        raise ValueError("live-refresh must be positive")
    if args.iteration_wait_ms < 0:
        raise ValueError("iteration-wait-ms cannot be negative")
    from quam_config import create_machine

    final_temperature = (
        args.temperature if args.final_temperature is None else args.final_temperature
    )
    problem = GridIsingProblem(args.rows, args.cols, args.coupling, args.field)
    initial_spins = np.random.default_rng(args.seed).choice(
        (-1, 1), size=problem.n_spins
    )
    machine = create_machine()
    qua_program = build_2d_pseudo_spin_program(
        machine,
        problem,
        args.iterations,
        args.temperature,
        final_temperature,
        initial_spins,
        args.seed,
        save_configurations=args.save_configurations or args.live_plot,
        iteration_wait_ns=round(args.iteration_wait_ms * 1_000_000),
    )
    qmm = machine.connect()
    qm = qmm.open_qm(machine.generate_config())

    started = time.perf_counter()
    job = qm.execute(qua_program)
    print(f"Submitted {args.rows}x{args.cols} 2D Ising experiment to real OPX hardware.")
    if args.live_plot:
        configurations, temperatures, total_magnetizations = fetch_2d_results_live(
            job,
            problem,
            args.iterations,
            args.live_refresh,
        )
    else:
        job.result_handles.wait_for_all_values()
        temperatures = _fetch_array(job, "temperatures").reshape(-1)
        total_magnetizations = _fetch_array(job, "magnetizations").reshape(-1)
        configurations = (
            _fetch_array(job, "configurations").reshape(-1, problem.n_spins)
            if args.save_configurations
            else None
        )
    duration = time.perf_counter() - started

    magnetizations = total_magnetizations / problem.n_spins
    output = save_metrics_csv(args.output, temperatures, magnetizations)
    print(f"Saved {len(temperatures)} iterations to {output}")
    print(f"Run time={duration:.3f} s")
    print(f"Final magnetization per spin={magnetizations[-1]:.6g}")

    if not args.no_plot:
        plot_2d_results(problem, temperatures, magnetizations, configurations)
        plt.show()


if __name__ == "__main__":
    main()
