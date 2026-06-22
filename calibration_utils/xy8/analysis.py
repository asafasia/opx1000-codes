import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import decay_exp
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


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
class XY8Fit:
    """Fitted XY8 coherence values for one qubit."""

    t2_xy8: dict[int, float]
    t2_xy8_error: dict[int, float]
    success: dict[int, bool]


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def _failed_fit(template: xr.DataArray) -> xr.DataArray:
    values = xr.DataArray(
        np.full(len(FIT_VALUES), np.nan),
        dims=("fit_vals",),
        coords={"fit_vals": FIT_VALUES},
    )
    for coord_name in ("qubit", "n_xy8"):
        if coord_name in template.coords:
            values = values.expand_dims({coord_name: [template.coords[coord_name].item()]})
    return values


def _fit_one_trace(trace: xr.DataArray, evolution_time: xr.DataArray, log_callable=None) -> xr.DataArray:
    x = np.asarray(evolution_time, dtype=float)
    y = np.asarray(trace, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    try:
        if np.count_nonzero(finite) < 4:
            raise ValueError("need at least four finite points")
        x_fit = x[finite]
        y_fit = y[finite]
        span = max(float(np.ptp(x_fit)), 1.0)
        initial = (
            float(y_fit[0] - y_fit[-1]),
            float(y_fit[-1]),
            -1 / span,
        )
        params, covariance = curve_fit(
            lambda t, a, offset, decay: decay_exp(t, a, offset, decay),
            x_fit,
            y_fit,
            p0=initial,
            maxfev=10000,
        )
    except Exception as error:
        if log_callable is not None:
            qubit = trace.coords.get("qubit", "?")
            n_xy8 = trace.coords.get("n_xy8", "?")
            log_callable(f"XY8 decay fit failed for qubit={qubit}, n_xy8={n_xy8}: {error}")
        return _failed_fit(trace)

    values = np.full(len(FIT_VALUES), np.nan, dtype=float)
    values[:3] = params
    if covariance.shape == (3, 3):
        values[3:] = covariance.reshape(-1)
    fit = xr.DataArray(
        values,
        dims=("fit_vals",),
        coords={"fit_vals": FIT_VALUES},
    )
    for coord_name in ("qubit", "n_xy8"):
        if coord_name in trace.coords:
            fit = fit.expand_dims({coord_name: [trace.coords[coord_name].item()]})
    return fit


def _fit_all(data: xr.DataArray, ds: xr.Dataset, log_callable=None) -> xr.DataArray:
    fits = []
    for qubit in data.qubit.values:
        qubit_fits = []
        for n_xy8 in data.n_xy8.values:
            trace = data.sel(qubit=qubit, n_xy8=n_xy8)
            fit = _fit_one_trace(trace, ds.evolution_time, log_callable=log_callable)
            qubit_fits.append(fit)
        fits.append(xr.concat(qubit_fits, dim=xr.IndexVariable("n_xy8", data.n_xy8.values)))
    return xr.concat(fits, dim=xr.IndexVariable("qubit", data.qubit.values))


def _fit_r_squared(data: xr.DataArray, fit: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    results = []
    for n_xy8 in data.n_xy8.values:
        selected_fit = fit.sel(n_xy8=n_xy8)
        fitted = decay_exp(
            ds.evolution_time,
            selected_fit.sel(fit_vals="a"),
            selected_fit.sel(fit_vals="offset"),
            selected_fit.sel(fit_vals="decay"),
        )
        selected_data = data.sel(n_xy8=n_xy8)
        residual_sum = ((selected_data - fitted) ** 2).sum(dim="evolution_time")
        total_sum = ((selected_data - selected_data.mean(dim="evolution_time")) ** 2).sum(dim="evolution_time")
        valid = np.isfinite(selected_fit).all(dim="fit_vals") & (
            selected_fit.sel(fit_vals="decay") < 0
        )
        results.append(xr.where(valid & (total_sum > 0), 1 - residual_sum / total_sum, np.nan))
    return xr.concat(results, dim=xr.IndexVariable("n_xy8", data.n_xy8.values))


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, XY8Fit]]:
    if node.parameters.use_state_discrimination:
        fit_data = _fit_all(ds.state, ds, node.log)
        selected_quadrature = xr.full_like(ds.state.isel(evolution_time=0, drop=True), "state", dtype="<U5")
    else:
        fit_i = _fit_all(ds.I, ds, node.log)
        fit_q = _fit_all(ds.Q, ds, node.log)
        r_squared_i = _fit_r_squared(ds.I, fit_i, ds)
        r_squared_q = _fit_r_squared(ds.Q, fit_q, ds)
        select_q = r_squared_q.notnull() & (r_squared_i.isnull() | (r_squared_q > r_squared_i))
        fit_data = xr.where(select_q, fit_q, fit_i)
        selected_quadrature = xr.where(select_q, "Q", "I")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    ds_fit = ds_fit.assign({"selected_quadrature": selected_quadrature})
    return _extract_relevant_fit_parameters(ds_fit)


def _extract_relevant_fit_parameters(fit: xr.Dataset):
    t2_xy8 = -1 / fit.fit_data.sel(fit_vals="decay")
    t2_xy8_error = -t2_xy8 * (
        np.sqrt(fit.fit_data.sel(fit_vals="decay_decay")) / fit.fit_data.sel(fit_vals="decay")
    )
    success = (
        np.isfinite(t2_xy8)
        & np.isfinite(t2_xy8_error)
        & (t2_xy8 > 16)
        & (t2_xy8_error >= 0)
        & (t2_xy8_error / t2_xy8 < 1)
    )

    fit = fit.assign(
        {
            "T2_xy8": t2_xy8,
            "T2_xy8_error": t2_xy8_error,
            "success": success,
        }
    )
    fit.T2_xy8.attrs = {"long_name": "T2 XY8", "units": "ns"}
    fit.T2_xy8_error.attrs = {"long_name": "T2 XY8 error", "units": "ns"}

    fit_results = {}
    for qubit in fit.qubit.values:
        selected = fit.sel(qubit=qubit)
        fit_results[str(qubit)] = XY8Fit(
            t2_xy8={
                int(n): float(selected.T2_xy8.sel(n_xy8=n))
                for n in fit.n_xy8.values
            },
            t2_xy8_error={
                int(n): float(selected.T2_xy8_error.sel(n_xy8=n))
                for n in fit.n_xy8.values
            },
            success={
                int(n): bool(selected.success.sel(n_xy8=n))
                for n in fit.n_xy8.values
            },
        )
    return fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit, result in fit_results.items():
        for n_xy8, t2 in result["t2_xy8"].items() if isinstance(result, dict) else result.t2_xy8.items():
            if isinstance(result, dict):
                error = result["t2_xy8_error"][n_xy8]
                success = result["success"][n_xy8]
            else:
                error = result.t2_xy8_error[n_xy8]
                success = result.success[n_xy8]
            status = "SUCCESS" if success else "FAIL"
            log_callable(
                f"XY8 T2 for {qubit}, N={n_xy8}: {1e-3 * t2:.2f} +/- {1e-3 * error:.2f} us --> {status}"
            )
