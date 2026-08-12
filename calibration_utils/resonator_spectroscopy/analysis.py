import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips
from calibration_utils.iq_blobs.analysis import _optimal_threshold


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    success: bool


def calculate_iq_separation(ds: xr.Dataset) -> xr.DataArray:
    """Return IQ-center distance divided by the pooled shot-cloud width."""
    pair_separations = _calculate_pairwise_iq_separations(ds)
    if pair_separations.sizes["state_pair"] > 1:
        separation = pair_separations.min(dim="state_pair")
    else:
        separation = pair_separations.isel(state_pair=0)
    separation.attrs = {
        "long_name": "IQ center separation / pooled standard deviation",
        "units": "",
    }
    return separation


def calculate_readout_fidelity(ds: xr.Dataset) -> xr.DataArray:
    """Return optimal threshold-discrimination fidelity versus frequency."""
    pairwise_fidelities = _calculate_pairwise_readout_fidelities(ds)
    if pairwise_fidelities.sizes["state_pair"] > 1:
        fidelity = pairwise_fidelities.min(dim="state_pair")
    else:
        fidelity = pairwise_fidelities.isel(state_pair=0)
    fidelity.attrs = {
        "long_name": "worst-pair optimal threshold discrimination fidelity",
        "units": "%",
    }
    return fidelity


def _binary_threshold_fidelity(
    first_i: np.ndarray,
    first_q: np.ndarray,
    second_i: np.ndarray,
    second_q: np.ndarray,
) -> float:
    """Calculate balanced assignment fidelity for two shot-level IQ clouds."""
    first_valid = np.isfinite(first_i) & np.isfinite(first_q)
    second_valid = np.isfinite(second_i) & np.isfinite(second_q)
    first_i = np.asarray(first_i[first_valid], dtype=float)
    first_q = np.asarray(first_q[first_valid], dtype=float)
    second_i = np.asarray(second_i[second_valid], dtype=float)
    second_q = np.asarray(second_q[second_valid], dtype=float)
    if first_i.size == 0 or second_i.size == 0:
        return np.nan

    delta_i = second_i.mean() - first_i.mean()
    delta_q = second_q.mean() - first_q.mean()
    angle = np.arctan2(-delta_q, delta_i)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    first_rotated = first_i * cosine - first_q * sine
    second_rotated = second_i * cosine - second_q * sine
    threshold = _optimal_threshold(first_rotated, second_rotated)
    if not np.isfinite(threshold):
        return np.nan

    first_correct = np.mean(first_rotated < threshold)
    second_correct = np.mean(second_rotated > threshold)
    return float(50.0 * (first_correct + second_correct))


def _calculate_pairwise_readout_fidelities(ds: xr.Dataset) -> xr.DataArray:
    states = _available_states(ds)
    if len(states) < 2:
        raise ValueError("Resonator spectroscopy requires at least two measured states.")
    pairs = [
        (f"{first}{second}", first, second)
        for first_index, first in enumerate(states)
        for second in states[first_index + 1 :]
    ]

    fidelities = []
    for pair_name, first, second in pairs:
        first_i, first_q = _state_iq(ds, first)
        second_i, second_q = _state_iq(ds, second)
        fidelity = xr.apply_ufunc(
            _binary_threshold_fidelity,
            first_i,
            first_q,
            second_i,
            second_q,
            input_core_dims=[["n_runs"]] * 4,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        fidelities.append(fidelity.expand_dims(state_pair=[pair_name]))

    pairwise_fidelities = xr.concat(fidelities, dim="state_pair")
    pairwise_fidelities.attrs = {
        "long_name": "pairwise optimal threshold discrimination fidelity",
        "units": "%",
    }
    return pairwise_fidelities


def _calculate_pairwise_iq_separations(ds: xr.Dataset) -> xr.DataArray:
    states = _available_states(ds)
    if len(states) < 2:
        raise ValueError("Resonator spectroscopy requires at least two measured states.")
    pairs = [
        (f"{first}{second}", first, second)
        for first_index, first in enumerate(states)
        for second in states[first_index + 1 :]
    ]

    separations = []
    for pair_name, first, second in pairs:
        first_I, first_Q = _state_iq(ds, first)
        second_I, second_Q = _state_iq(ds, second)
        center_distance = np.hypot(
            second_I.mean(dim="n_runs") - first_I.mean(dim="n_runs"),
            second_Q.mean(dim="n_runs") - first_Q.mean(dim="n_runs"),
        )
        first_width = np.sqrt(first_I.var(dim="n_runs") + first_Q.var(dim="n_runs"))
        second_width = np.sqrt(second_I.var(dim="n_runs") + second_Q.var(dim="n_runs"))
        pooled_width = np.sqrt((first_width**2 + second_width**2) / 2)
        separations.append(
            xr.where(pooled_width > 0, center_distance / pooled_width, np.nan)
            .expand_dims(state_pair=[pair_name])
        )

    pair_separations = xr.concat(separations, dim="state_pair")
    pair_separations.attrs = {
        "long_name": "pairwise IQ center separation / pooled standard deviation",
        "units": "",
    }
    return pair_separations


def _state_iq(ds: xr.Dataset, state: str) -> tuple[xr.DataArray, xr.DataArray]:
    state_variables = {
        "g": ("Ig", "Qg"),
        "e": ("Im", "Qm"),
        "f": ("If", "Qf"),
    }
    i_name, q_name = state_variables[state]
    return ds[i_name], ds[q_name]


def _available_states(ds: xr.Dataset) -> list[str]:
    state_variables = {
        "g": ("Ig", "Qg"),
        "e": ("Im", "Qm"),
        "f": ("If", "Qf"),
    }
    return [
        state
        for state, variables in state_variables.items()
        if set(variables).issubset(ds.data_vars)
    ]


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
        s_freq = f"\tMaximum-separation readout frequency: {1e-9 * fit_results[q]['frequency']:.6f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    state_labels = {"g": "ground", "e": "mixed", "f": "f"}
    states = _available_states(ds)
    iq_list = [
        variable
        for state in states
        for variable in {
            "g": ("Ig", "Qg"),
            "e": ("Im", "Qm"),
            "f": ("If", "Qf"),
        }[state]
    ]
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=iq_list)
    if "n_runs" not in ds.dims:
        raise ValueError(
            "Resonator spectroscopy requires individual shots along the 'n_runs' dimension."
        )

    for state in states:
        label = state_labels[state]
        state_I, state_Q = _state_iq(ds, state)
        state_data = add_amplitude_and_phase(
            xr.Dataset(
                {
                    "I": state_I.mean(dim="n_runs"),
                    "Q": state_Q.mean(dim="n_runs"),
                }
            ),
            "detuning",
            subtract_slope_flag=True,
        )
        ds[f"{label}_IQ_abs"] = state_data.IQ_abs
        ds[f"{label}_phase"] = state_data.phase
    ds["pairwise_IQ_separation"] = _calculate_pairwise_iq_separations(ds)
    ds["IQ_separation"] = calculate_iq_separation(ds)
    ds["pairwise_readout_fidelity"] = _calculate_pairwise_readout_fidelities(ds)
    ds["readout_fidelity"] = calculate_readout_fidelity(ds)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    fit_source = _fit_source_iq_abs(ds)
    fit_results = peaks_dips(fit_source, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


def _fit_source_iq_abs(ds: xr.Dataset) -> xr.DataArray:
    for variable in ("ground_IQ_abs", "mixed_IQ_abs", "f_IQ_abs"):
        if variable in ds.data_vars:
            return ds[variable]
    raise ValueError("Resonator spectroscopy data does not contain a fitted state response.")


def _extract_relevant_fit_parameters(
    fit: xr.Dataset, spectroscopy_data: xr.Dataset, node: QualibrationNode
):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Choose the readout frequency that maximizes normalized state separation.
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    separation_detuning = spectroscopy_data.detuning.isel(
        detuning=spectroscopy_data.IQ_separation.argmax(dim="detuning")
    )
    res_freq = separation_detuning + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {
        "long_name": "maximum-separation readout frequency",
        "units": "Hz",
    }
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
