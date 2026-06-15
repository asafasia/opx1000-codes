"""Classical model and result utilities for amplitude-encoded pseudo-spins."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPIN_DOWN_AMPLITUDE = 0.5
SPIN_UP_AMPLITUDE = 1.0
DEFAULT_OBSERVABLE_WINDOW = 20


def correlation_length(configurations: np.ndarray) -> float:
    """Estimate the connected-correlation decay length for a 1D spin ring."""
    configurations = np.asarray(configurations, dtype=float)
    n_spins = configurations.shape[1]
    magnetization = float(np.mean(configurations))
    distances = np.arange(1, n_spins // 2 + 1)
    connected = np.asarray(
        [
            np.mean(configurations * np.roll(configurations, -distance, axis=1))
            - magnetization**2
            for distance in distances
        ]
    )
    nonpositive = np.flatnonzero(connected <= 1e-8)
    fit_limit = int(nonpositive[0]) if len(nonpositive) else len(connected)
    if fit_limit < 2:
        return 0.0
    slope, _ = np.polyfit(
        distances[:fit_limit],
        np.log(connected[:fit_limit]),
        1,
    )
    return min(float(-1.0 / slope), n_spins / 2) if slope < 0 else 0.0


def rolling_observables(
    configurations: np.ndarray,
    energies: np.ndarray,
    magnetizations: np.ndarray,
    temperatures: np.ndarray,
    window: int = DEFAULT_OBSERVABLE_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rolling heat-capacity, susceptibility, and correlation estimates."""
    n_spins = configurations.shape[1]
    heat_capacities = []
    susceptibilities = []
    correlation_lengths = []
    for iteration in range(len(configurations)):
        start = max(0, iteration + 1 - window)
        sample_energies = energies[start : iteration + 1]
        sample_magnetizations = magnetizations[start : iteration + 1]
        sample_configurations = configurations[start : iteration + 1]
        heat_capacities.append(
            np.var(sample_energies) / (n_spins * temperatures[iteration] ** 2)
        )
        susceptibilities.append(
            np.var(sample_magnetizations) / (n_spins * temperatures[iteration])
        )
        correlation_lengths.append(correlation_length(sample_configurations))
    return (
        np.asarray(heat_capacities),
        np.asarray(susceptibilities),
        np.asarray(correlation_lengths),
    )


def ring_couplings(n_spins: int, coupling: float) -> np.ndarray:
    """Return a nearest-neighbor ring coupling matrix."""
    if n_spins < 2:
        raise ValueError("n_spins must be at least 2")
    matrix = np.zeros((n_spins, n_spins), dtype=float)
    for spin in range(n_spins):
        neighbor = (spin + 1) % n_spins
        matrix[spin, neighbor] = coupling
        matrix[neighbor, spin] = coupling
    return matrix


@dataclass(frozen=True)
class IsingProblem:
    """An arbitrary Ising problem with E = -1/2 s.T J s - h.T s."""

    couplings: np.ndarray
    fields: np.ndarray

    def __post_init__(self) -> None:
        couplings = np.asarray(self.couplings, dtype=float)
        fields = np.asarray(self.fields, dtype=float)
        if couplings.ndim != 2 or couplings.shape[0] != couplings.shape[1]:
            raise ValueError("couplings must be a square matrix")
        if couplings.shape[0] < 1:
            raise ValueError("problem must contain at least one spin")
        if fields.shape != (couplings.shape[0],):
            raise ValueError("fields must contain one value per spin")
        if not np.all(np.isfinite(couplings)) or not np.all(np.isfinite(fields)):
            raise ValueError("couplings and fields must be finite")
        if not np.allclose(couplings, couplings.T):
            raise ValueError("couplings must be symmetric")
        if not np.allclose(np.diag(couplings), 0):
            raise ValueError("couplings must have a zero diagonal")
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "fields", fields)

    @property
    def n_spins(self) -> int:
        return len(self.fields)

    def validate_spins(self, spins: np.ndarray) -> np.ndarray:
        spins = np.asarray(spins, dtype=np.int8)
        if spins.shape != (self.n_spins,) or not np.all(np.isin(spins, (-1, 1))):
            raise ValueError(f"spins must have shape ({self.n_spins},) and values -1/+1")
        return spins

    def energy(self, spins: np.ndarray) -> float:
        spins = self.validate_spins(spins)
        return float(-0.5 * spins @ self.couplings @ spins - self.fields @ spins)

    def magnetization(self, spins: np.ndarray) -> int:
        return int(np.sum(self.validate_spins(spins)))

    def delta_energy(self, spins: np.ndarray, spin_index: int) -> float:
        spins = self.validate_spins(spins)
        local_field = self.fields[spin_index] + self.couplings[spin_index] @ spins
        return float(2.0 * spins[spin_index] * local_field)

    def amplitudes(self, spins: np.ndarray) -> np.ndarray:
        spins = self.validate_spins(spins)
        return np.where(spins == 1, SPIN_UP_AMPLITUDE, SPIN_DOWN_AMPLITUDE)


@dataclass(frozen=True)
class IsingRun:
    """Saved state and observables from a pseudo-spin experiment."""

    configurations: np.ndarray
    temperatures: np.ndarray
    accepted_flips: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray | None = None
    heat_capacities: np.ndarray | None = None
    susceptibilities: np.ndarray | None = None
    correlation_lengths: np.ndarray | None = None

    @classmethod
    def from_lightweight_metrics(
        cls,
        temperatures: np.ndarray,
        magnetizations: np.ndarray,
        n_spins: int,
    ) -> "IsingRun":
        """Create a run without storing or analyzing spin configurations."""
        temperatures = np.asarray(temperatures, dtype=float).reshape(-1)
        magnetizations = np.asarray(magnetizations, dtype=float).reshape(-1)
        if magnetizations.shape != temperatures.shape:
            raise ValueError("magnetizations must contain one value per temperature")
        return cls(
            configurations=np.empty((len(temperatures), 0), dtype=np.int8),
            temperatures=temperatures,
            accepted_flips=np.empty(0, dtype=int),
            magnetizations=magnetizations / n_spins,
        )

    @classmethod
    def from_configurations(
        cls,
        problem: IsingProblem,
        configurations: np.ndarray,
        temperatures: np.ndarray,
        accepted_flips: np.ndarray,
        full_analysis: bool = False,
    ) -> "IsingRun":
        configurations = np.asarray(configurations, dtype=np.int8)
        temperatures = np.asarray(temperatures, dtype=float).reshape(-1)
        accepted_flips = np.asarray(accepted_flips, dtype=int).reshape(-1)
        expected_shape = (len(temperatures), problem.n_spins)
        if configurations.shape != expected_shape:
            raise ValueError(f"configurations must have shape {expected_shape}")
        if accepted_flips.shape != (len(temperatures),):
            raise ValueError("accepted_flips must contain one value per iteration")
        total_magnetizations = np.sum(configurations, axis=1, dtype=int)
        magnetizations = total_magnetizations / problem.n_spins
        energies = None
        heat_capacities = None
        susceptibilities = None
        correlation_lengths = None
        if full_analysis:
            total_energies = np.asarray(
                [problem.energy(spins) for spins in configurations]
            )
            energies = total_energies / problem.n_spins
            heat_capacities, susceptibilities, correlation_lengths = (
                rolling_observables(
                    configurations,
                    total_energies,
                    total_magnetizations,
                    temperatures,
                )
            )
        return cls(
            configurations=configurations,
            temperatures=temperatures,
            accepted_flips=accepted_flips,
            energies=energies,
            magnetizations=magnetizations,
            heat_capacities=heat_capacities,
            susceptibilities=susceptibilities,
            correlation_lengths=correlation_lengths,
        )

    def save_csv(self, path: Path, problem: IsingProblem) -> Path:
        """Save lightweight iteration metrics or full per-spin analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if self.energies is None:
                writer.writerow(
                    (
                        "iteration",
                        "temperature",
                        "magnetization_per_spin",
                    )
                )
                for iteration in range(len(self.temperatures)):
                    writer.writerow(
                        (
                            iteration,
                            self.temperatures[iteration],
                            self.magnetizations[iteration],
                        )
                    )
                return path

            writer.writerow(
                (
                    "iteration",
                    "spin_index",
                    "spin",
                    "pulse_amplitude",
                    "temperature",
                    "energy_per_spin",
                    "magnetization_per_spin",
                    "accepted_flips",
                    "heat_capacity_per_spin",
                    "susceptibility_per_spin",
                    "correlation_length",
                )
            )
            for iteration, configuration in enumerate(self.configurations):
                amplitudes = problem.amplitudes(configuration)
                for spin_index, (spin, amplitude) in enumerate(
                    zip(configuration, amplitudes)
                ):
                    writer.writerow(
                        (
                            iteration,
                            spin_index,
                            spin,
                            amplitude,
                            self.temperatures[iteration],
                            self.energies[iteration],
                            self.magnetizations[iteration],
                            self.accepted_flips[iteration],
                            self.heat_capacities[iteration],
                            self.susceptibilities[iteration],
                            self.correlation_lengths[iteration],
                        )
                    )
        return path

    def plot(self):
        """Plot lightweight metrics or the complete analysis."""
        if self.energies is None:
            return self.plot_lightweight()
        return self.plot_full_analysis()

    def plot_lightweight(self):
        """Plot only temperature and magnetization per spin."""
        iterations = np.arange(len(self.configurations))
        fig, magnetization_axis = plt.subplots(figsize=(11, 5))
        temperature_axis = magnetization_axis.twinx()
        magnetization_line = magnetization_axis.plot(
            iterations,
            self.magnetizations,
            label="magnetization per spin",
            color="tab:blue",
        )[0]
        temperature_line = temperature_axis.plot(
            iterations,
            self.temperatures,
            label="temperature",
            color="tab:red",
        )[0]
        magnetization_axis.set_xlabel("Iteration")
        magnetization_axis.set_ylabel("Magnetization per spin", color="tab:blue")
        temperature_axis.set_ylabel("Temperature", color="tab:red")
        magnetization_axis.tick_params(axis="y", colors="tab:blue")
        temperature_axis.tick_params(axis="y", colors="tab:red")
        magnetization_axis.grid(alpha=0.25)
        magnetization_axis.legend(
            handles=(magnetization_line, temperature_line),
            loc="upper right",
        )
        fig.tight_layout()
        return fig

    def plot_full_analysis(self):
        """Plot spin evolution, intensive metrics, and rolling observables."""
        iterations = np.arange(len(self.configurations))
        fig = plt.figure(figsize=(11, 9.5))
        grid = fig.add_gridspec(
            4,
            2,
            width_ratios=(1, 0.035),
            height_ratios=(1.3, 1, 1, 0.22),
            hspace=0.12,
            wspace=0.08,
        )
        spin_axis = fig.add_subplot(grid[0, 0])
        metric_axis = fig.add_subplot(grid[1, 0], sharex=spin_axis)
        observable_axis = fig.add_subplot(grid[2, 0], sharex=spin_axis)
        colorbar_axis = fig.add_subplot(grid[0, 1])
        legend_axis = fig.add_subplot(grid[3, :])
        legend_axis.axis("off")
        for row in (1, 2):
            spacer_axis = fig.add_subplot(grid[row, 1])
            spacer_axis.axis("off")

        image = spin_axis.imshow(
            self.configurations.T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        spin_axis.set_ylabel("Pseudo-spin")
        spin_axis.set_title("Amplitude-Encoded Spin Evolution")
        fig.colorbar(image, cax=colorbar_axis, label="Spin")

        energy_line = metric_axis.plot(
            iterations,
            self.energies,
            label="energy per spin",
        )[0]
        magnetization_line = metric_axis.plot(
            iterations,
            self.magnetizations,
            label="magnetization per spin",
        )[0]
        metric_axis.set_ylabel("Per-spin value")
        metric_axis.grid(alpha=0.25)

        heat_line = observable_axis.plot(
            iterations,
            self.heat_capacities,
            label="heat capacity per spin",
            linestyle="--",
            color="tab:green",
        )[0]
        susceptibility_line = observable_axis.plot(
            iterations,
            self.susceptibilities,
            label="susceptibility per spin",
            linestyle=":",
            color="tab:orange",
        )[0]
        correlation_axis = observable_axis.twinx()
        correlation_line = correlation_axis.plot(
            iterations,
            self.correlation_lengths,
            label="correlation length",
            linestyle="--",
            color="tab:purple",
        )[0]
        legend_axis.legend(
            handles=(
                energy_line,
                magnetization_line,
                heat_line,
                susceptibility_line,
                correlation_line,
            ),
            loc="center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.15),
        )
        observable_axis.set_ylabel("Heat capacity / susceptibility")
        correlation_axis.set_ylabel("Correlation length", color="tab:purple")
        correlation_axis.tick_params(axis="y", colors="tab:purple")
        observable_axis.grid(alpha=0.25)
        observable_axis.set_xlabel("Iteration")

        plt.setp(spin_axis.get_xticklabels(), visible=False)
        plt.setp(metric_axis.get_xticklabels(), visible=False)
        return fig


def metropolis_reference(
    problem: IsingProblem,
    iterations: int,
    temperature_start: float,
    temperature_end: float | None = None,
    seed: int | None = None,
    initial_spins: np.ndarray | None = None,
    full_analysis: bool = False,
) -> IsingRun:
    """Run the same random-site Metropolis update used by the QUA experiment."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    temperature_end = temperature_start if temperature_end is None else temperature_end
    if temperature_start <= 0 or temperature_end <= 0:
        raise ValueError("temperatures must be positive")

    rng = np.random.default_rng(seed)
    spins = (
        rng.choice((-1, 1), size=problem.n_spins).astype(np.int8)
        if initial_spins is None
        else problem.validate_spins(initial_spins).copy()
    )
    temperatures = np.linspace(temperature_start, temperature_end, iterations)
    configurations = []
    accepted_flips = []

    for temperature in temperatures:
        accepted = 0
        for _ in range(problem.n_spins):
            spin_index = int(rng.integers(problem.n_spins))
            delta = problem.delta_energy(spins, spin_index)
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                spins[spin_index] *= -1
                accepted += 1
        configurations.append(spins.copy())
        accepted_flips.append(accepted)

    return IsingRun.from_configurations(
        problem,
        np.asarray(configurations),
        temperatures,
        np.asarray(accepted_flips),
        full_analysis=full_analysis,
    )
