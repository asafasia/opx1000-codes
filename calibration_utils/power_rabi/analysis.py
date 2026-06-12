import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    success: bool
    selected_quadrature: str
    r_squared_I: float
    r_squared_Q: float


def _target_amplitude_prefactor(frequency, number_of_pulses: int, operation: str):
    """Convert a Rabi oscillation frequency into the selected operation amplitude."""
    if operation.endswith("x180"):
        rotation_fraction = 0.25
    elif operation.endswith("x90") or operation.endswith("y90"):
        rotation_fraction = 0.125
    else:
        raise ValueError(f"Unsupported Rabi operation {operation!r}.")
    return number_of_pulses * rotation_fraction / abs(frequency)


def _r_squared(data: xr.DataArray, fit: xr.DataArray, dim: str) -> xr.DataArray:
    """Return R-squared, using NaN for constant data or invalid fits."""
    fitted_data = oscillation(
        data[dim],
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="f"),
        fit.sel(fit_vals="phi"),
        fit.sel(fit_vals="offset"),
    )
    residual_sum = ((data - fitted_data) ** 2).sum(dim=dim)
    total_sum = ((data - data.mean(dim=dim)) ** 2).sum(dim=dim)
    fit_is_valid = np.isfinite(fit).all(dim="fit_vals") & (abs(fit.sel(fit_vals="f")) > 0)
    return xr.where(fit_is_valid & (total_sum > 0), 1 - residual_sum / total_sum, np.nan)


def _select_best_quadrature_fit(
    data_I: xr.DataArray,
    data_Q: xr.DataArray,
    fit_I: xr.DataArray,
    fit_Q: xr.DataArray,
    dim: str,
):
    """Select the I or Q oscillation fit with the highest valid R-squared."""
    r_squared_I = _r_squared(data_I, fit_I, dim)
    r_squared_Q = _r_squared(data_Q, fit_Q, dim)
    select_Q = r_squared_Q.notnull() & (r_squared_I.isnull() | (r_squared_Q > r_squared_I))
    selected_fit = xr.where(select_Q, fit_Q, fit_I)
    selected_quadrature = xr.where(select_Q, "Q", "I")
    return selected_fit, r_squared_I, r_squared_Q, selected_quadrature


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
        s_amp = (
            f"The calibrated {fit_results[q]['operation']} amplitude: "
            f"{1e3 * fit_results[q]['opt_amp']:.2f} mV "
            f"(x{fit_results[q]['opt_amp_prefactor']:.2f}, "
            f"selected {fit_results[q]['selected_quadrature']})\n "
        )
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    if node.name == "13_power_rabi_ef":
        full_amp = np.array([ds.amp_prefactor * q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]])
    else:
        full_amp = np.array(
            [ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]]
        )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    if node.name == "13_power_rabi_ef" and hasattr(ds, "I"):
        ds = add_amplitude_and_phase(ds, "amp_prefactor", subtract_slope_flag=True)
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
    operation = getattr(node.parameters, "operation", "EF_x180" if node.name == "13_power_rabi_ef" else "x180")
    is_1d_scan = "nb_of_pulses" not in ds.dims or ds.sizes["nb_of_pulses"] == 1
    if is_1d_scan:
        ds_fit = ds.isel(nb_of_pulses=0, drop=True) if "nb_of_pulses" in ds.dims else ds
        # Fit the power Rabi oscillations
        if node.parameters.use_state_discrimination:
            fit_vals = fit_oscillation(ds_fit.state, "amp_prefactor")
            ds_fit = xr.merge([ds, fit_vals.rename("fit")])
        else:
            fit_I = fit_oscillation(ds_fit.I, "amp_prefactor")
            fit_Q = fit_oscillation(ds_fit.Q, "amp_prefactor")
            fit_vals, r_squared_I, r_squared_Q, selected_quadrature = _select_best_quadrature_fit(
                ds_fit.I,
                ds_fit.Q,
                fit_I,
                fit_Q,
                "amp_prefactor",
            )
            ds_fit = xr.merge(
                [
                    ds,
                    fit_vals.rename("fit"),
                    fit_I.rename("fit_I"),
                    fit_Q.rename("fit_Q"),
                    r_squared_I.rename("r_squared_I"),
                    r_squared_Q.rename("r_squared_Q"),
                    selected_quadrature.rename("selected_quadrature"),
                ]
            )
    else:
        ds_fit = ds
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if node.parameters.use_state_discrimination:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (ds.nb_of_pulses.data[0] % 2 == 0 and operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 != 0 and operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    operation = getattr(node.parameters, "operation", "EF_x180" if node.name == "13_power_rabi_ef" else "x180")
    is_1d_scan = "nb_of_pulses" not in fit.dims or fit.sizes["nb_of_pulses"] == 1
    if is_1d_scan:
        # The fitted frequency gives the complete Rabi period. Deriving the
        # target rotation from the period avoids selecting the wrong extremum
        # when the fitted phase is shifted.
        number_of_pulses = int(fit.nb_of_pulses.values[0]) if "nb_of_pulses" in fit.coords else 1
        factor = _target_amplitude_prefactor(
            fit.fit.sel(fit_vals="f"),
            number_of_pulses,
            operation,
        )
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": f"factor to get an {operation} pulse",
            "units": "",
        }
        if node.name == "13_power_rabi_ef":
            current_amps = xr.DataArray(
                [q.xy.operations["EF_x180"].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        else:
            current_amps = xr.DataArray(
                [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
                coords=dict(qubit=fit.qubit.data),
            )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    else:
        current_amps = xr.DataArray(
            [q.xy.operations[operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    amp_success = fit.opt_amp < limits[0].max_x180_wf_amplitude
    success_criteria = ~nan_success & amp_success
    if is_1d_scan and "selected_quadrature" in fit:
        success_criteria &= np.isfinite(fit.r_squared_I) | np.isfinite(fit.r_squared_Q)
    fit = fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.__float__(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.__float__(),
            operation=operation,
            success=fit.sel(qubit=q).success.values.__bool__(),
            selected_quadrature=(
                str(fit.sel(qubit=q).selected_quadrature.values)
                if "selected_quadrature" in fit
                else ("state" if node.parameters.use_state_discrimination else "I")
            ),
            r_squared_I=(
                float(fit.sel(qubit=q).r_squared_I.values) if "r_squared_I" in fit else float("nan")
            ),
            r_squared_Q=(
                float(fit.sel(qubit=q).r_squared_Q.values) if "r_squared_Q" in fit else float("nan")
            ),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
