"""Correctness checks for the amplitude-encoded pseudo-spin model."""

import numpy as np

from ising_machine.model import IsingProblem, IsingRun, metropolis_reference, ring_couplings
from ising_machine.pseudo_spin_experiment import uniform_ring_parameters


def test_amplitude_mapping() -> None:
    problem = IsingProblem(ring_couplings(4, 0.25), np.zeros(4))
    amplitudes = problem.amplitudes(np.asarray([-1, 1, -1, 1]))
    assert np.array_equal(amplitudes, [0.5, 1.0, 0.5, 1.0])


def test_delta_energy_matches_energy_change() -> None:
    problem = IsingProblem(
        ring_couplings(5, 0.4),
        np.asarray([0.1, 0.0, -0.1, 0.2, 0.0]),
    )
    spins = np.asarray([-1, 1, -1, -1, 1], dtype=np.int8)
    before = problem.energy(spins)
    expected_delta = problem.delta_energy(spins, 3)
    spins[3] *= -1
    assert np.isclose(problem.energy(spins) - before, expected_delta)


def test_reference_saves_every_iteration() -> None:
    problem = IsingProblem(ring_couplings(10, 0.25), np.zeros(10))
    run = metropolis_reference(
        problem,
        iterations=25,
        temperature_start=1.0,
        seed=3,
        full_analysis=True,
    )
    assert run.configurations.shape == (25, 10)
    assert run.energies.shape == (25,)
    assert run.magnetizations.shape == (25,)
    assert run.heat_capacities.shape == (25,)
    assert run.susceptibilities.shape == (25,)
    assert run.correlation_lengths.shape == (25,)
    assert np.all(np.isfinite(run.heat_capacities))
    assert np.all(np.isfinite(run.susceptibilities))
    assert np.all(np.isfinite(run.correlation_lengths))
    assert np.all(run.heat_capacities >= 0)
    assert np.all(run.susceptibilities >= 0)
    assert np.all(run.correlation_lengths >= 0)
    assert np.all(np.abs(run.magnetizations) <= 1)
    assert np.all(np.isin(run.configurations, (-1, 1)))
    assert np.all((run.accepted_flips >= 0) & (run.accepted_flips <= 10))


def test_lightweight_reference_skips_heavy_analysis() -> None:
    problem = IsingProblem(ring_couplings(10, 0.25), np.zeros(10))
    run = metropolis_reference(problem, iterations=5, temperature_start=1.0, seed=3)
    assert run.energies is None
    assert run.heat_capacities is None
    assert run.susceptibilities is None
    assert run.correlation_lengths is None
    assert run.magnetizations.shape == (5,)


def test_controller_metric_run_has_no_configurations() -> None:
    run = IsingRun.from_lightweight_metrics(
        temperatures=np.asarray([0.7, 0.7]),
        magnetizations=np.asarray([10, -4]),
        n_spins=20,
    )
    assert run.configurations.shape == (2, 0)
    assert np.array_equal(run.magnetizations, [0.5, -0.2])


def test_uniform_ring_detection() -> None:
    ring = IsingProblem(ring_couplings(100, 0.25), np.full(100, 0.1))
    assert uniform_ring_parameters(ring) == (0.25, 0.1)

    nonuniform = IsingProblem(ring.couplings, np.linspace(0.0, 0.1, 100))
    assert uniform_ring_parameters(nonuniform) is None


if __name__ == "__main__":
    test_amplitude_mapping()
    test_delta_energy_matches_energy_change()
    test_reference_saves_every_iteration()
    test_lightweight_reference_skips_heavy_analysis()
    test_controller_metric_run_has_no_configurations()
    test_uniform_ring_detection()
    print("Pseudo-spin Ising checks passed.")
