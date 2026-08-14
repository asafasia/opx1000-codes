"""Joint thermal-relaxation analysis for ground/excited T1 traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.constants import h, k
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class ThermalRelaxationFit:
    """Joint ground/excited relaxation fit and derived temperature."""

    t1_ns: float
    t1_error_ns: float
    equilibrium_excited_population: float
    equilibrium_excited_population_error: float
    initial_ground_trace_population: float
    initial_excited_trace_population: float
    qubit_frequency_hz: float
    effective_temperature_kelvin: float
    effective_temperature_error_kelvin: float


def mitigated_state_trace(dataset: xr.Dataset, qubit_name: str) -> xr.DataArray:
    """Return a readout-mitigated excited-state population trace."""
    selected = dataset.sel(qubit=qubit_name)
    if "state" not in selected:
        raise ValueError(
            "Effective-temperature analysis requires state discrimination."
        )
    if not selected.state.attrs.get("readout_mitigated", False):
        raise ValueError(
            "Effective-temperature analysis requires readout-mitigated state data."
        )
    return selected.state


def thermal_relaxation_model(
    coordinates: np.ndarray,
    t1_ns: float,
    equilibrium_excited_population: float,
    initial_ground_trace_population: float,
    initial_excited_trace_population: float,
) -> np.ndarray:
    """Population model with a shared T1 and thermal equilibrium population."""
    idle_time_ns, initial_state_code = np.asarray(coordinates, dtype=float)
    initial_population = np.where(
        initial_state_code < 0.5,
        initial_ground_trace_population,
        initial_excited_trace_population,
    )
    return equilibrium_excited_population + (
        initial_population - equilibrium_excited_population
    ) * np.exp(-idle_time_ns / t1_ns)


def effective_temperature_from_population(
    equilibrium_excited_population: float,
    qubit_frequency_hz: float,
) -> float:
    """Return the two-level effective temperature in kelvin."""
    population = float(equilibrium_excited_population)
    if not 0.0 < population < 1.0:
        raise ValueError(
            "Equilibrium excited-state population must be between 0 and 1."
        )
    log_population_ratio = np.log((1.0 - population) / population)
    if np.isclose(log_population_ratio, 0.0):
        return float(np.inf)
    return float(h * qubit_frequency_hz / (k * log_population_ratio))


def _temperature_error_from_population_error(
    equilibrium_excited_population: float,
    population_error: float,
    qubit_frequency_hz: float,
) -> float:
    """Propagate the fitted population uncertainty to temperature."""
    population = float(equilibrium_excited_population)
    log_population_ratio = np.log((1.0 - population) / population)
    if np.isclose(log_population_ratio, 0.0):
        return float(np.inf)
    derivative = (
        h
        * qubit_frequency_hz
        / (k * population * (1.0 - population) * log_population_ratio**2)
    )
    return float(abs(derivative) * population_error)


def fit_thermal_relaxation(
    ground: xr.Dataset,
    excited: xr.Dataset,
    *,
    qubit_name: str,
    qubit_frequency_hz: float,
) -> ThermalRelaxationFit:
    """Jointly fit the g/e traces and derive the effective temperature."""
    ground_trace = mitigated_state_trace(ground, qubit_name)
    excited_trace = mitigated_state_trace(excited, qubit_name)
    ground_times = np.asarray(ground_trace.idle_time.values, dtype=float)
    excited_times = np.asarray(excited_trace.idle_time.values, dtype=float)
    ground_population = np.asarray(ground_trace.values, dtype=float).reshape(-1)
    excited_population = np.asarray(excited_trace.values, dtype=float).reshape(-1)

    if ground_times.size < 4 or excited_times.size < 4:
        raise ValueError("At least four idle-time points are required for each trace.")
    if not (
        np.all(np.isfinite(ground_population))
        and np.all(np.isfinite(excited_population))
    ):
        raise ValueError("T1 traces contain non-finite populations.")

    idle_times = np.concatenate([ground_times, excited_times])
    initial_state_codes = np.concatenate(
        [np.zeros_like(ground_times), np.ones_like(excited_times)]
    )
    populations = np.concatenate([ground_population, excited_population])

    tail_points = max(3, min(ground_times.size, excited_times.size) // 5)
    equilibrium_guess = float(
        np.mean(
            np.concatenate(
                [ground_population[-tail_points:], excited_population[-tail_points:]]
            )
        )
    )
    maximum_idle_time = float(max(np.max(ground_times), np.max(excited_times)))
    initial_guess = (
        max(maximum_idle_time / 4.0, 4.0),
        float(np.clip(equilibrium_guess, 1e-6, 1.0 - 1e-6)),
        float(np.clip(ground_population[0], 0.0, 1.0)),
        float(np.clip(excited_population[0], 0.0, 1.0)),
    )
    fit_parameters, covariance = curve_fit(
        thermal_relaxation_model,
        np.vstack([idle_times, initial_state_codes]),
        populations,
        p0=initial_guess,
        bounds=(
            (1e-9, 1e-9, 0.0, 0.0),
            (np.inf, 1.0 - 1e-9, 1.0, 1.0),
        ),
        maxfev=50_000,
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t1_ns, equilibrium_population, initial_ground, initial_excited = fit_parameters
    t1_error_ns, equilibrium_error = errors[:2]
    temperature_kelvin = effective_temperature_from_population(
        equilibrium_population,
        qubit_frequency_hz,
    )
    temperature_error_kelvin = _temperature_error_from_population_error(
        equilibrium_population,
        equilibrium_error,
        qubit_frequency_hz,
    )
    return ThermalRelaxationFit(
        t1_ns=float(t1_ns),
        t1_error_ns=float(t1_error_ns),
        equilibrium_excited_population=float(equilibrium_population),
        equilibrium_excited_population_error=float(equilibrium_error),
        initial_ground_trace_population=float(initial_ground),
        initial_excited_trace_population=float(initial_excited),
        qubit_frequency_hz=float(qubit_frequency_hz),
        effective_temperature_kelvin=temperature_kelvin,
        effective_temperature_error_kelvin=temperature_error_kelvin,
    )
