import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    success: bool


def calculate_iq_separation(ds: xr.Dataset) -> xr.DataArray:
    """Return IQ-center distance divided by the pooled shot-cloud width."""
    ground_I = ds.Ig.mean(dim="n_runs")
    ground_Q = ds.Qg.mean(dim="n_runs")
    mixed_I = ds.Im.mean(dim="n_runs")
    mixed_Q = ds.Qm.mean(dim="n_runs")
    center_distance = np.hypot(mixed_I - ground_I, mixed_Q - ground_Q)
    ground_width = np.sqrt(ds.Ig.var(dim="n_runs") + ds.Qg.var(dim="n_runs"))
    mixed_width = np.sqrt(ds.Im.var(dim="n_runs") + ds.Qm.var(dim="n_runs"))
    pooled_width = np.sqrt((ground_width**2 + mixed_width**2) / 2)
    separation = xr.where(pooled_width > 0, center_distance / pooled_width, np.nan)
    separation.attrs = {
        "long_name": "IQ center separation / pooled standard deviation",
        "units": "",
    }
    return separation


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
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Im", "Qm"])
    if "n_runs" not in ds.dims:
        raise ValueError(
            "Resonator spectroscopy requires individual shots along the 'n_runs' dimension."
        )

    ground_I = ds.Ig.mean(dim="n_runs")
    ground_Q = ds.Qg.mean(dim="n_runs")
    mixed_I = ds.Im.mean(dim="n_runs")
    mixed_Q = ds.Qm.mean(dim="n_runs")
    ground = add_amplitude_and_phase(
        xr.Dataset({"I": ground_I, "Q": ground_Q}),
        "detuning",
        subtract_slope_flag=True,
    )
    mixed = add_amplitude_and_phase(
        xr.Dataset({"I": mixed_I, "Q": mixed_Q}),
        "detuning",
        subtract_slope_flag=True,
    )
    ds["ground_IQ_abs"] = ground.IQ_abs
    ds["ground_phase"] = ground.phase
    ds["mixed_IQ_abs"] = mixed.IQ_abs
    ds["mixed_phase"] = mixed.phase
    ds["IQ_separation"] = calculate_iq_separation(ds)
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
    # Fit the resonator line
    fit_results = peaks_dips(ds.ground_IQ_abs, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


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
