import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


def _spectroscopy_center_frequency(qubit, node_name: str) -> float:
    """Return the configured transition frequency around which the experiment sweeps."""
    if node_name == "12_qubit_spectroscopy_EF":
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
    full_freq = np.array(
        [ds.detuning + _spectroscopy_center_frequency(q, node.name) for q in node.namespace["qubits"]]
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
    ds_fit = xr.merge([ds_fit, fit_vals])
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array(
        [_spectroscopy_center_frequency(q, node.name) for q in node.namespace["qubits"]]
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
