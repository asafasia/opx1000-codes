import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips

MIN_FIT_R_SQUARED = 0.8


def _spectroscopy_center_frequency(qubit, transition: str) -> float:
    """Return the configured transition frequency around which the experiment sweeps."""
    if transition == "ef":
        if qubit.f_12 is not None:
            return float(qubit.f_12)
        return float(qubit.f_01 - qubit.anharmonicity)
    return float(qubit.xy.RF_frequency)


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    relative_freq: float
    fwhm: float
    iw_angle: float
    saturation_amp: float
    x180_amp: float
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
        s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        s_angle = f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
        s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
        s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    transition = getattr(node.parameters, "transition", "ge")
    full_freq = np.array(
        [
            ds.detuning + _spectroscopy_center_frequency(q, transition)
            for q in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

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
    ds_fit = ds
    if node.parameters.use_state_discrimination:
        fit_signal = ds_fit.state
        ds_fit = ds_fit.assign(
            {
                "iw_angle": xr.DataArray(
                    [
                        q.resonator.operations["readout"].integration_weights_angle
                        for q in node.namespace["qubits"]
                    ],
                    coords={"qubit": ds_fit.qubit},
                )
            }
        )
    else:
        shifts = np.abs((ds_fit.IQ_abs - ds_fit.IQ_abs.mean(dim="detuning"))).idxmax(dim="detuning")
        angle = np.arctan2(
            ds_fit.sel(detuning=shifts).Q - ds_fit.Q.mean(dim="detuning"),
            ds_fit.sel(detuning=shifts).I - ds_fit.I.mean(dim="detuning"),
        )
        ds_fit = ds_fit.assign({"iw_angle": angle})
        ds_fit = ds_fit.assign(
            {"I_rot": ds_fit.I * np.cos(ds_fit.iw_angle) + ds_fit.Q * np.sin(ds_fit.iw_angle)}
        )
        fit_signal = ds_fit.I_rot

    fit_vals = peaks_dips(fit_signal, dim="detuning", prominence_factor=5)
    max_fit_vals = _fit_maximum_with_measured_fallback(fit_signal)
    fit_vals = xr.merge([fit_vals, max_fit_vals])
    fit_vals = fit_vals.assign(
        {
            "position": xr.where(
                fit_vals.fit_r_squared >= MIN_FIT_R_SQUARED,
                fit_vals.fit_position,
                fit_vals.measured_max_position,
            ),
            "width": xr.where(
                fit_vals.fit_r_squared >= MIN_FIT_R_SQUARED,
                fit_vals.fit_width,
                fit_vals.width,
            ),
        }
    )
    ds_fit = xr.merge([ds_fit, fit_vals])
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _gaussian_peak(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _fit_maximum_with_measured_fallback(fit_signal: xr.DataArray) -> xr.Dataset:
    qubits = list(fit_signal.qubit.values)
    measured_positions = []
    fit_positions = []
    fit_widths = []
    fit_scores = []
    fit_offsets = []
    fit_amplitudes = []
    fit_sigmas = []

    for qubit in qubits:
        trace = fit_signal.sel(qubit=qubit)
        (
            measured_position,
            fit_position,
            fit_width,
            fit_score,
            fit_offset,
            fit_amplitude,
            fit_sigma,
        ) = _fit_trace_maximum(
            np.asarray(trace.detuning.values, dtype=float),
            np.asarray(trace.values, dtype=float),
        )
        measured_positions.append(measured_position)
        fit_positions.append(fit_position)
        fit_widths.append(fit_width)
        fit_scores.append(fit_score)
        fit_offsets.append(fit_offset)
        fit_amplitudes.append(fit_amplitude)
        fit_sigmas.append(fit_sigma)

    return xr.Dataset(
        {
            "measured_max_position": ("qubit", measured_positions),
            "fit_position": ("qubit", fit_positions),
            "fit_width": ("qubit", fit_widths),
            "fit_r_squared": ("qubit", fit_scores),
            "fit_offset": ("qubit", fit_offsets),
            "fit_amplitude": ("qubit", fit_amplitudes),
            "fit_sigma": ("qubit", fit_sigmas),
        },
        coords={"qubit": qubits},
    )


def _fit_trace_maximum(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 5 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    max_index = int(np.argmax(y))
    measured_position = float(x[max_index])
    span = float(np.ptp(x))
    step = _median_step(x)
    baseline = float(np.percentile(y, 10))
    amplitude = max(float(y[max_index] - baseline), np.finfo(float).eps)
    sigma = max(span / 10, step)

    try:
        params, _ = curve_fit(
            _gaussian_peak,
            x,
            y,
            p0=[baseline, amplitude, measured_position, sigma],
            bounds=(
                [-np.inf, 0.0, float(np.min(x)), max(step / 10, np.finfo(float).eps)],
                [np.inf, np.inf, float(np.max(x)), span],
            ),
            maxfev=10000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return measured_position, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    offset, amplitude, center, sigma = [float(value) for value in params]
    sigma = abs(sigma)
    if (
        not all(np.isfinite(value) for value in (offset, amplitude, center, sigma))
        or sigma <= 0
    ):
        return measured_position, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    fitted = _gaussian_peak(x, offset, amplitude, center, sigma)
    residual_sum_squares = float(np.sum((y - fitted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    if total_sum_squares <= 0:
        return measured_position, center, np.nan, np.nan, offset, amplitude, sigma

    r_squared = 1 - residual_sum_squares / total_sum_squares
    fwhm = float(2 * np.sqrt(2 * np.log(2)) * sigma)
    return measured_position, center, fwhm, float(r_squared), offset, amplitude, sigma


def _median_step(x: np.ndarray) -> float:
    steps = np.diff(np.sort(np.unique(x)))
    if steps.size == 0:
        return 1.0
    return max(float(np.median(np.abs(steps))), np.finfo(float).eps)


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    transition = getattr(node.parameters, "transition", "ge")
    full_freq = np.array(
        [
            _spectroscopy_center_frequency(q, transition)
            for q in node.namespace["qubits"]
        ]
    )
    res_freq = fit.position + full_freq
    rel_freq = fit.position
    fit = fit.assign({"res_freq": ("qubit", res_freq.data)})
    fit = fit.assign({"relative_freq": ("qubit", rel_freq.data)})
    fit.res_freq.attrs = {"long_name": "qubit xy frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign({"fwhm": fwhm})
    fit.fwhm.attrs = {"long_name": "qubit fwhm", "units": "Hz"}
    # State-discriminated data already uses the configured angle.
    if not node.parameters.use_state_discrimination:
        prev_angles = np.array(
            [q.resonator.operations["readout"].integration_weights_angle for q in node.namespace["qubits"]]
        )
        fit = fit.assign({"iw_angle": (prev_angles + fit.iw_angle) % (2 * np.pi)})
    fit.iw_angle.attrs = {"long_name": "integration weight angle", "units": "rad"}
    # Get saturation amplitude
    x180_length = np.array([q.xy.operations["x180"].length * 1e-9 for q in node.namespace["qubits"]])
    used_amp = np.array(
        [
            q.xy.operations["saturation"].amplitude * node.parameters.operation_amplitude_factor
            for q in node.namespace["qubits"]
        ]
    )
    factor_cw = node.parameters.target_peak_width / fit.width
    fit = fit.assign({"saturation_amplitude": factor_cw * used_amp / node.parameters.operation_amplitude_factor})
    # get expected x180 amplitude
    factor_x180 = np.pi / (fit.width * x180_length)
    fit = fit.assign({"x180_amplitude": factor_x180 * used_amp})

    # Assess whether the fit was successful or not. Keep this permissive:
    # spectroscopy should fail only when no usable resonance was fitted, not
    # because the derived pulse-amplitude suggestion is aggressive.
    half_span_hz = 0.5 * node.parameters.frequency_span_in_mhz * 1e6
    edge_margin_hz = max(node.parameters.frequency_step_in_mhz * 1e6, 0.02 * half_span_hz)
    freq_success = np.isfinite(rel_freq) & (np.abs(rel_freq) <= half_span_hz + edge_margin_hz)
    fwhm_success = np.isfinite(fwhm) & (fwhm > 0)
    success_criteria = freq_success & fwhm_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            relative_freq=fit.sel(qubit=q).relative_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            iw_angle=fit.sel(qubit=q).iw_angle.values.__float__(),
            saturation_amp=fit.sel(qubit=q).saturation_amplitude.values.__float__(),
            x180_amp=fit.sel(qubit=q).x180_amplitude.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
