import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import oscillation_decay_exp


FIT_VALUES = [
    "a",
    "f",
    "phi",
    "offset",
    "decay",
    *[
        f"{left}_{right}"
        for left in ("a", "f", "phi", "offset", "decay")
        for right in ("a", "f", "phi", "offset", "decay")
    ],
]


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    freq_offset: float
    decay: float
    decay_error: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_detuning = f"\tDetuning to correct: {1e-6 * fit_results[q]['freq_offset']:.3f} MHz | "
        s_T2 = f"T2*: {1e6 * fit_results[q]['decay']:.1f} µs\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_detuning + s_T2)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    data = ds.state if node.parameters.use_state_discrimination else ds.I
    frequency_guess_per_ns = abs(float(node.parameters.frequency_detuning_in_mhz)) * 1e-3
    fit = _fit_ramsey_with_frequency_guess(data, "idle_time", frequency_guess_per_ns)

    ds_fit = xr.merge([ds, fit.rename("fit")])

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _fit_ramsey_with_frequency_guess(
    data: xr.DataArray,
    dim: str,
    frequency_guess_per_ns: float,
) -> xr.DataArray:
    """Fit Ramsey traces robustly, trying the configured detuning frequency first."""

    def fit_trace(time, signal):
        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)
        valid = np.isfinite(time) & np.isfinite(signal)
        time = time[valid]
        signal = signal[valid]
        if time.size < 6 or np.ptp(time) <= 0 or np.ptp(signal) <= 0:
            return np.full(len(FIT_VALUES), np.nan)

        order = np.argsort(time)
        time = time[order]
        signal = signal[order]
        positive_steps = np.diff(time)
        positive_steps = positive_steps[positive_steps > 0]
        if positive_steps.size == 0:
            return np.full(len(FIT_VALUES), np.nan)

        max_frequency = 0.5 / float(np.min(positive_steps))
        configured_frequency = float(np.clip(frequency_guess_per_ns, np.finfo(float).eps, max_frequency))
        frequency_starts = [configured_frequency]
        fft_frequency = _fft_frequency_guess(time, signal)
        if np.isfinite(fft_frequency) and 0 < fft_frequency <= max_frequency:
            if not np.isclose(fft_frequency, configured_frequency):
                frequency_starts.append(float(fft_frequency))

        decay_guess = 1 / float(np.ptp(time))
        best_result = None
        best_residual = np.inf
        for frequency_start in frequency_starts:
            amplitude_guess, phase_guess, offset_guess = _sinusoid_initial_values(
                time,
                signal,
                frequency_start,
            )
            initial_values = [
                amplitude_guess,
                frequency_start,
                phase_guess,
                offset_guess,
                decay_guess,
            ]
            try:
                fitted_values, covariance = curve_fit(
                    oscillation_decay_exp,
                    time,
                    signal,
                    p0=initial_values,
                    bounds=(
                        [-np.inf, 0, -np.inf, -np.inf, 0],
                        [np.inf, max_frequency, np.inf, np.inf, np.inf],
                    ),
                    method="trf",
                    loss="soft_l1",
                    max_nfev=50_000,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue

            residual = float(np.sum((signal - oscillation_decay_exp(time, *fitted_values)) ** 2))
            if np.isfinite(residual) and residual < best_residual:
                best_residual = residual
                best_result = np.concatenate([fitted_values, np.asarray(covariance).reshape(-1)])

        if best_result is None or best_result.size != len(FIT_VALUES):
            return np.full(len(FIT_VALUES), np.nan)
        return best_result

    fit = xr.apply_ufunc(
        fit_trace,
        data[dim],
        data,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"fit_vals": len(FIT_VALUES)}},
    )
    return fit.assign_coords(fit_vals=("fit_vals", FIT_VALUES))


def _fft_frequency_guess(time: np.ndarray, signal: np.ndarray) -> float:
    """Estimate a fallback positive frequency after interpolation to a uniform grid."""
    uniform_time = np.linspace(float(time[0]), float(time[-1]), time.size)
    uniform_signal = np.interp(uniform_time, time, signal)
    centered = uniform_signal - np.mean(uniform_signal)
    frequencies = np.fft.rfftfreq(time.size, d=float(uniform_time[1] - uniform_time[0]))
    amplitudes = np.abs(np.fft.rfft(centered))
    positive = frequencies > 0
    if not np.any(positive):
        return np.nan
    return float(frequencies[positive][np.argmax(amplitudes[positive])])


def _sinusoid_initial_values(
    time: np.ndarray,
    signal: np.ndarray,
    frequency: float,
) -> tuple[float, float, float]:
    """Estimate amplitude, phase, and offset for a fixed starting frequency."""
    angle = 2 * np.pi * frequency * time
    design = np.column_stack([np.cos(angle), np.sin(angle), np.ones_like(angle)])
    cosine, sine, offset = np.linalg.lstsq(design, signal, rcond=None)[0]
    amplitude = float(np.hypot(cosine, sine))
    phase = float(np.arctan2(-sine, cosine))
    return amplitude, phase, float(offset)


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add calculated metadata to the dataset
    frequency = fit.sel(fit_vals="f")
    frequency.attrs = {"long_name": "frequency", "units": "MHz"}
    frequency = frequency.where(frequency > 0, drop=True)

    decay = fit.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    decay_res = fit.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay residual", "units": "nSec"}

    tau = 1 / decay
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    tau_error = tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T2* error", "units": "uSec"}

    detuning = int(node.parameters.frequency_detuning_in_mhz * 1e6)

    freq_offset, decay, decay_error = calculate_fit_results(frequency, tau, tau_error, fit, detuning)
    # Assess whether the fit was successful or not
    nan_success = np.isnan(freq_offset.fit) | np.isnan(decay.fit)
    success_criteria = ~nan_success
    fit = fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            freq_offset=1e9 * float(freq_offset.sel(qubit=q).fit),
            decay=float(decay.sel(qubit=q).fit),
            decay_error=float(decay_error.sel(qubit=q).fit),
            success=bool(fit.sel(qubit=q).success.values),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results


def calculate_fit_results(frequency, tau, tau_error, fit, detuning):
    """
    Calculate fit results such as frequency offset, decay, and decay error.

    Parameters:
        frequency (xarray.DataArray): Frequency data.
        tau (xarray.DataArray): Tau values.
        tau_error (xarray.DataArray): Tau error values.
        fit (xarray.DataArray): Fit results.
        detuning (float): Detuning parameter in Hz.

    Returns:
        tuple: Frequency offset, decay, and decay error.
    """
    within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="detuning_signs") == 1
    positive_shift = frequency.sel(detuning_signs=1) > frequency.sel(detuning_signs=-1)
    freq_offset = (
        within_detuning * (frequency * fit.detuning_signs).mean(dim="detuning_signs")
        + ~within_detuning * positive_shift * frequency.mean(dim="detuning_signs")
        - ~within_detuning * ~positive_shift * frequency.mean(dim="detuning_signs")
    )

    decay = 1e-9 * tau.mean(dim="detuning_signs")
    decay_error = 1e-9 * tau_error.mean(dim="detuning_signs")

    return freq_offset, decay, decay_error
