"""Two-dimensional Ising model and a Metropolis Monte Carlo solver."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np


@dataclass
class IsingModel:
    """Square-lattice Ising model with periodic boundary conditions.

    The Hamiltonian is

        E = -J * sum_<i,j> s_i s_j - h * sum_i s_i,

    where every spin is either -1 or +1.
    """

    size: int
    coupling: float = 1.0
    field: float = 0.0
    seed: int | None = None
    ordered: bool = False
    spins: np.ndarray = dataclass_field(init=False, repr=False)
    rng: np.random.Generator = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        self.rng = np.random.default_rng(self.seed)
        if self.ordered:
            self.spins = np.ones((self.size, self.size), dtype=np.int8)
        else:
            self.spins = self.rng.choice((-1, 1), size=(self.size, self.size)).astype(
                np.int8
            )

    @property
    def n_spins(self) -> int:
        return self.size * self.size

    def total_energy(self) -> float:
        """Return energy while counting each nearest-neighbor bond once."""
        right = np.roll(self.spins, -1, axis=1)
        down = np.roll(self.spins, -1, axis=0)
        interaction = -self.coupling * np.sum(self.spins * (right + down))
        field_energy = -self.field * np.sum(self.spins)
        return float(interaction + field_energy)

    def magnetization(self) -> int:
        """Return the total magnetization."""
        return int(np.sum(self.spins))

    def delta_energy(self, row: int, col: int) -> float:
        """Return the energy change produced by flipping one spin."""
        spin = self.spins[row, col]
        neighbor_sum = (
            self.spins[(row - 1) % self.size, col]
            + self.spins[(row + 1) % self.size, col]
            + self.spins[row, (col - 1) % self.size]
            + self.spins[row, (col + 1) % self.size]
        )
        return float(2.0 * spin * (self.coupling * neighbor_sum + self.field))

    def sweep(self, temperature: float) -> float:
        """Attempt one random spin flip per lattice site.

        Returns:
            Fraction of proposed flips that were accepted.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        accepted = 0
        for _ in range(self.n_spins):
            row = int(self.rng.integers(self.size))
            col = int(self.rng.integers(self.size))
            delta = self.delta_energy(row, col)
            if delta <= 0 or self.rng.random() < np.exp(-delta / temperature):
                self.spins[row, col] *= -1
                accepted += 1
        return accepted / self.n_spins

    def run(
        self,
        temperature: float,
        equilibration_sweeps: int,
        sample_sweeps: int,
        sample_interval: int = 1,
    ) -> dict[str, np.ndarray]:
        """Equilibrate and collect energy, magnetization, and spin samples."""
        if (
            equilibration_sweeps < 0
            or sample_sweeps < 1
            or sample_interval < 1
            or sample_interval > sample_sweeps
        ):
            raise ValueError("invalid sweep counts or sample interval")

        for _ in range(equilibration_sweeps):
            self.sweep(temperature)

        energies = []
        magnetizations = []
        configurations = []
        acceptance_rates = []

        for sweep_index in range(sample_sweeps):
            acceptance_rates.append(self.sweep(temperature))
            if (sweep_index + 1) % sample_interval == 0:
                energies.append(self.total_energy())
                magnetizations.append(self.magnetization())
                configurations.append(self.spins.copy())

        return {
            "energies": np.asarray(energies, dtype=float),
            "magnetizations": np.asarray(magnetizations, dtype=float),
            "configurations": np.asarray(configurations, dtype=np.int8),
            "acceptance_rates": np.asarray(acceptance_rates, dtype=float),
        }
