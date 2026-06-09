"""Offline checks for the 2D amplitude-encoded Ising experiment."""

import numpy as np

from .pseudo_spin_2d_experiment import GridIsingProblem


def test_periodic_neighbors() -> None:
    problem = GridIsingProblem(3, 4)
    left, right, up, down = problem.neighbor_indices()
    assert (left[0], right[0], up[0], down[0]) == (3, 1, 8, 4)


def test_ordered_grid_energy() -> None:
    problem = GridIsingProblem(4, 5, coupling=0.25)
    spins = np.ones(problem.n_spins, dtype=np.int8)
    assert problem.energy_per_spin(spins) == -0.5
    assert problem.delta_energy(spins, 0) == 2.0


def test_delta_energy_matches_total_change() -> None:
    problem = GridIsingProblem(4, 5, coupling=0.25, field=0.1)
    spins = np.random.default_rng(3).choice((-1, 1), size=problem.n_spins)
    before = problem.energy_per_spin(spins) * problem.n_spins
    delta = problem.delta_energy(spins, 7)
    spins[7] *= -1
    after = problem.energy_per_spin(spins) * problem.n_spins
    assert np.isclose(after - before, delta)


if __name__ == "__main__":
    test_periodic_neighbors()
    test_ordered_grid_energy()
    test_delta_energy_matches_total_change()
    print("2D pseudo-spin Ising checks passed.")
