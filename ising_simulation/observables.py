"""Thermodynamic and spatial observables for Ising samples."""

from __future__ import annotations

import numpy as np


def axis_correlation(configurations: np.ndarray) -> np.ndarray:
    """Return <s(0)s(r)> averaged over samples, sites, and both lattice axes."""
    size = configurations.shape[-1]
    correlations = []
    for distance in range(size // 2 + 1):
        horizontal = configurations * np.roll(configurations, -distance, axis=2)
        vertical = configurations * np.roll(configurations, -distance, axis=1)
        correlations.append(0.5 * (horizontal.mean() + vertical.mean()))
    return np.asarray(correlations)


def correlation_length(correlation: np.ndarray, magnetization_per_spin: float) -> float:
    """Estimate a connected-correlation decay length from an exponential fit."""
    connected = correlation - magnetization_per_spin**2
    distances = np.arange(len(connected))
    valid = (distances > 0) & (connected > 1e-8)
    if np.count_nonzero(valid) < 2:
        return 0.0
    slope, _ = np.polyfit(distances[valid], np.log(connected[valid]), 1)
    return float(-1.0 / slope) if slope < 0 else 0.0


def summarize_samples(
    samples: dict[str, np.ndarray], temperature: float, n_spins: int
) -> dict[str, float]:
    """Calculate intensive observables from a Monte Carlo sample."""
    energies = samples["energies"]
    magnetizations = samples["magnetizations"]
    configurations = samples["configurations"]
    energy_variance = float(np.var(energies))
    magnetization_variance = float(np.var(magnetizations))
    mean_magnetization = float(np.mean(magnetizations) / n_spins)
    correlation = axis_correlation(configurations)

    return {
        "temperature": temperature,
        "mean_energy_per_spin": float(np.mean(energies) / n_spins),
        "energy_variance_per_spin": energy_variance / n_spins,
        "mean_magnetization_per_spin": mean_magnetization,
        "mean_abs_magnetization_per_spin": float(
            np.mean(np.abs(magnetizations)) / n_spins
        ),
        "magnetization_variance_per_spin": magnetization_variance / n_spins,
        "heat_capacity_per_spin": energy_variance / (n_spins * temperature**2),
        "susceptibility_per_spin": magnetization_variance / (n_spins * temperature),
        "nearest_neighbor_correlation": float(correlation[1]),
        "correlation_length": correlation_length(correlation, mean_magnetization),
        "mean_acceptance_rate": float(np.mean(samples["acceptance_rates"])),
    }
