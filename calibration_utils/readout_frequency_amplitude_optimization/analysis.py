import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Best point from the 2D readout frequency/amplitude map."""

    optimal_frequency: float
    optimal_detuning: float
    optimal_amp_prefactor: float
    optimal_amplitude: float
    readout_fidelity: float
    state_difference: float
    separation_to_width: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tOptimal RF: {result['optimal_frequency'] * 1e-9:.6f} GHz "
            f"({result['optimal_detuning'] * 1e-6:+.2f} MHz) | "
            f"amp prefactor: {result['optimal_amp_prefactor']:.3f} | "
            f"readout amp: {result['optimal_amplitude'] * 1e3:.2f} mV | "
            f"fidelity: {result['readout_fidelity']:.1f} % | "
            f"diff: {result['state_difference'] * 1e3:.3f} mV | "
            f"sep/width: {result['separation_to_width']:.2f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    readout_amplitude = np.array(
        [
            ds.amp_prefactor * q.resonator.operations["readout"].amplitude
            for q in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq),
        readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitude),
    )
    ds.full_freq.attrs = {"long_name": "Readout RF frequency", "units": "Hz"}
    ds.readout_amplitude.attrs = {"long_name": "Readout amplitude", "units": "V"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    ds_fit = ds.copy()
    delta_i = ds_fit.Ie.mean(dim="n_runs") - ds_fit.Ig.mean(dim="n_runs")
    delta_q = ds_fit.Qe.mean(dim="n_runs") - ds_fit.Qg.mean(dim="n_runs")
    state_difference = np.hypot(delta_i, delta_q)
    ground_width = np.sqrt(ds_fit.Ig.var(dim="n_runs") + ds_fit.Qg.var(dim="n_runs"))
    excited_width = np.sqrt(ds_fit.Ie.var(dim="n_runs") + ds_fit.Qe.var(dim="n_runs"))
    pooled_width = np.sqrt((ground_width**2 + excited_width**2) / 2)
    separation_to_width = xr.where(pooled_width > 0, state_difference / pooled_width, np.nan)
    ds_fit = ds_fit.assign(
        {
            "state_difference": state_difference,
            "separation_to_width": separation_to_width,
        }
    )

    fidelity, threshold, angle = _compute_fidelity_maps(ds_fit)
    ds_fit = ds_fit.assign(
        {
            "readout_fidelity": fidelity,
            "ge_threshold": threshold,
            "iw_angle": angle,
        }
    )
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _compute_fidelity_maps(ds: xr.Dataset):
    qubits = ds.qubit.values
    detunings = ds.detuning.values
    amps = ds.amp_prefactor.values
    fidelity = np.full((len(qubits), len(detunings), len(amps)), np.nan)
    threshold = np.full_like(fidelity, np.nan)
    angle = np.full_like(fidelity, np.nan)

    for qi, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit)
        for di, detuning in enumerate(detunings):
            for ai, amp in enumerate(amps):
                point = selected.sel(detuning=detuning, amp_prefactor=amp)
                ground_i = np.asarray(point.Ig, dtype=float)
                ground_q = np.asarray(point.Qg, dtype=float)
                excited_i = np.asarray(point.Ie, dtype=float)
                excited_q = np.asarray(point.Qe, dtype=float)
                delta_i = np.nanmean(excited_i) - np.nanmean(ground_i)
                delta_q = np.nanmean(excited_q) - np.nanmean(ground_q)
                point_angle = np.arctan2(-delta_q, delta_i)
                c = np.cos(point_angle)
                s = np.sin(point_angle)
                ground_rot = ground_i * c - ground_q * s
                excited_rot = excited_i * c - excited_q * s
                point_threshold, point_fidelity = _optimal_threshold_and_fidelity(
                    ground_rot,
                    excited_rot,
                )
                fidelity[qi, di, ai] = 100 * point_fidelity
                threshold[qi, di, ai] = point_threshold
                angle[qi, di, ai] = point_angle

    coords = {"qubit": qubits, "detuning": detunings, "amp_prefactor": amps}
    dims = ("qubit", "detuning", "amp_prefactor")
    return (
        xr.DataArray(fidelity, dims=dims, coords=coords),
        xr.DataArray(threshold, dims=dims, coords=coords),
        xr.DataArray(angle, dims=dims, coords=coords),
    )


def _optimal_threshold_and_fidelity(ground, excited):
    ground = np.asarray(ground, dtype=float)
    excited = np.asarray(excited, dtype=float)
    ground = ground[np.isfinite(ground)]
    excited = excited[np.isfinite(excited)]
    if ground.size == 0 or excited.size == 0:
        return np.nan, np.nan

    values = np.concatenate((ground, excited))
    labels = np.concatenate(
        (np.zeros(ground.size, dtype=np.int8), np.ones(excited.size, dtype=np.int8))
    )
    unique_values, inverse = np.unique(values, return_inverse=True)
    ground_counts = np.bincount(inverse[labels == 0], minlength=unique_values.size)
    excited_counts = np.bincount(inverse[labels == 1], minlength=unique_values.size)

    cumulative_ground = np.cumsum(ground_counts)
    cumulative_excited = np.cumsum(excited_counts)
    errors = np.concatenate(
        ([ground.size], ground.size - cumulative_ground + cumulative_excited)
    )
    best_index = int(np.argmin(errors))
    if best_index == 0:
        threshold = float(np.nextafter(unique_values[0], -np.inf))
    elif best_index == unique_values.size:
        threshold = float(np.nextafter(unique_values[-1], np.inf))
    else:
        threshold = float(0.5 * (unique_values[best_index - 1] + unique_values[best_index]))

    ground_correct = np.mean(ground < threshold)
    excited_correct = np.mean(excited > threshold)
    return threshold, float((ground_correct + excited_correct) / 2)


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    fit_results = {}
    success = []
    for q in node.namespace["qubits"]:
        selected = ds_fit.sel(qubit=q.name)
        if np.isfinite(selected.readout_fidelity).any():
            flat_index = int(np.nanargmax(selected.readout_fidelity.values))
            det_index, amp_index = np.unravel_index(
                flat_index,
                selected.readout_fidelity.shape,
            )
            optimal_detuning = float(selected.detuning.values[det_index])
            optimal_amp_prefactor = float(selected.amp_prefactor.values[amp_index])
            optimal_frequency = float(selected.full_freq.isel(detuning=det_index))
            optimal_amplitude = float(selected.readout_amplitude.isel(amp_prefactor=amp_index))
            readout_fidelity = float(selected.readout_fidelity.values[det_index, amp_index])
            state_difference = float(selected.state_difference.values[det_index, amp_index])
            separation_to_width = float(selected.separation_to_width.values[det_index, amp_index])
            is_success = bool(np.isfinite(readout_fidelity) and readout_fidelity > 50)
        else:
            optimal_frequency = np.nan
            optimal_detuning = np.nan
            optimal_amp_prefactor = np.nan
            optimal_amplitude = np.nan
            readout_fidelity = np.nan
            state_difference = np.nan
            separation_to_width = np.nan
            is_success = False
        success.append(is_success)
        fit_results[q.name] = FitParameters(
            optimal_frequency=optimal_frequency,
            optimal_detuning=optimal_detuning,
            optimal_amp_prefactor=optimal_amp_prefactor,
            optimal_amplitude=optimal_amplitude,
            readout_fidelity=readout_fidelity,
            state_difference=state_difference,
            separation_to_width=separation_to_width,
            success=is_success,
        )

    ds_fit = ds_fit.assign(
        {"success": xr.DataArray(success, dims="qubit", coords={"qubit": ds_fit.qubit.values})}
    )
    return ds_fit, fit_results
