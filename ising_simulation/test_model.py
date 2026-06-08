"""Small correctness checks runnable without a test framework."""

import numpy as np

from .model import IsingModel
from .observables import summarize_samples


def test_ordered_lattice_energy() -> None:
    model = IsingModel(size=4, ordered=True)
    assert model.total_energy() == -2 * model.n_spins
    assert model.magnetization() == model.n_spins
    assert model.delta_energy(0, 0) == 8.0


def test_flip_energy_matches_total_energy_change() -> None:
    model = IsingModel(size=5, seed=3)
    energy_before = model.total_energy()
    expected_change = model.delta_energy(2, 4)
    model.spins[2, 4] *= -1
    assert np.isclose(model.total_energy() - energy_before, expected_change)


def test_summary_is_finite() -> None:
    model = IsingModel(size=6, seed=1)
    samples = model.run(temperature=2.5, equilibration_sweeps=10, sample_sweeps=20)
    summary = summarize_samples(samples, temperature=2.5, n_spins=model.n_spins)
    assert all(np.isfinite(value) for value in summary.values())


if __name__ == "__main__":
    test_ordered_lattice_energy()
    test_flip_energy_matches_total_energy_change()
    test_summary_is_finite()
    print("Ising simulation checks passed.")
