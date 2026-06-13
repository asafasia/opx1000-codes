import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Tuple
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import decay_exp, fit_decay_exp


FIT_VALUES = [
    "a",
    "offset",
    "decay",
    "a_a",
    "a_offset",
    "a_decay",
    "offset_a",
    "offset_offset",
    "offset_decay",
    "decay_a",
    "decay_offset",
    "decay_decay",
]


@dataclass
class T1Fit:
    """Stores the relevant T1 experiment fit parameters for a single qubit"""

    t1: float
    t1_error: float
    success: bool


def log_fitted_results(ds: xr.Dataset, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
        Expected variables: 'tau', 'tau_error', 'success'.
        Expected coordinate: 'qubit'.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    Returns:
    --------
    None

    Example:
    --------
        >>> log_fitted_results(ds)
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in ds.qubit.values:
        if ds.sel(qubit=q).success.values:
            log_callable(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> SUCCESS!"
            )
        else:
            log_callable(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> FAIL!"
            )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def _failed_fit(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Create a NaN fit with the same non-sweep dimensions as the data."""
    template = data.isel({dim: 0}, drop=True)
    return xr.concat(
        [xr.full_like(template, np.nan, dtype=float) for _ in FIT_VALUES],
        dim=xr.IndexVariable("fit_vals", FIT_VALUES),
    ).transpose(..., "fit_vals")


def _fit_decay_per_qubit(data: xr.DataArray, dim: str, log_callable=None) -> xr.DataArray:
    """Fit each qubit independently so one failed fit does not abort the node."""
    fits = []
    for qubit in data.qubit.values:
        qubit_data = data.sel(qubit=[qubit])
        try:
            fit = fit_decay_exp(qubit_data, dim)
        except Exception as error:
            if log_callable is not None:
                log_callable(f"T1 decay fit failed for {qubit}: {error}")
            fit = _failed_fit(qubit_data, dim)
        fits.append(fit)
    return xr.concat(fits, dim="qubit")


def _fit_r_squared(data: xr.DataArray, fit: xr.DataArray, dim: str) -> xr.DataArray:
    """Return the goodness of an exponential fit, or NaN for an invalid fit."""
    fitted = decay_exp(
        data[dim],
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="offset"),
        fit.sel(fit_vals="decay"),
    )
    residual_sum = ((data - fitted) ** 2).sum(dim=dim)
    total_sum = ((data - data.mean(dim=dim)) ** 2).sum(dim=dim)
    valid = np.isfinite(fit).all(dim="fit_vals") & (fit.sel(fit_vals="decay") < 0)
    return xr.where(valid & (total_sum > 0), 1 - residual_sum / total_sum, np.nan)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, T1Fit]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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

    # Fit the exponential decay. The library fitter can return None for one
    # failed trace, so fit qubits independently and preserve the raw data.
    if node.parameters.use_state_discrimination:
        fit_data = _fit_decay_per_qubit(ds.state, "idle_time", node.log)
        selected_quadrature = xr.full_like(ds.state.isel(idle_time=0, drop=True), "state", dtype="<U5")
    else:
        fit_I = _fit_decay_per_qubit(ds.I, "idle_time", node.log)
        fit_Q = _fit_decay_per_qubit(ds.Q, "idle_time", node.log)
        r_squared_I = _fit_r_squared(ds.I, fit_I, "idle_time")
        r_squared_Q = _fit_r_squared(ds.Q, fit_Q, "idle_time")
        select_Q = r_squared_Q.notnull() & (r_squared_I.isnull() | (r_squared_Q > r_squared_I))
        fit_data = xr.where(select_Q, fit_Q, fit_I)
        selected_quadrature = xr.where(select_Q, "Q", "I")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    ds_fit = ds_fit.assign({"selected_quadrature": selected_quadrature})
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit)

    return fit_data, fit_results


def _fit_t1_with_exponential_decay(ds, use_state_discrimination):
    """Perform the fitting process based on the state discrimination flag."""
    if use_state_discrimination:
        fit = fit_decay_exp(ds.state, "idle_time")
    else:
        fit = fit_decay_exp(ds.I, "idle_time")
    return fit


def _extract_relevant_fit_parameters(fit: xr.Dataset):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "time", "units": "ns"}
    # Get the fitted T1
    tau = -1 / fit.fit_data.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.data))
    fit.tau.attrs = {"long_name": "T1", "units": "ns"}
    # Get the error on T1
    tau_error = -tau * (np.sqrt(fit.fit_data.sel(fit_vals="decay_decay")) / fit.fit_data.sel(fit_vals="decay"))
    fit = fit.assign_coords(tau_error=("qubit", tau_error.data))
    fit.tau_error.attrs = {"long_name": "T1 error", "units": "ns"}
    # Assess whether the fit was successful or not
    success_criteria = (
        np.isfinite(tau.data)
        & np.isfinite(tau_error.data)
        & (tau.data > 16)
        & (tau_error.data >= 0)
        & (tau_error.data / tau.data < 1)
    )
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: T1Fit(
            t1=fit.sel(qubit=q).tau.values.__float__(),
            t1_error=fit.sel(qubit=q).tau_error.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
