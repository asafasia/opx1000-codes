import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


KDE_GRID_SIZE = 80
KDE_PROBABILITY = 0.95
STATE_SPECS = (
    ("g", "Ig", "Qg", "ground", "Ground"),
    ("e", "Ie", "Qe", "prepared", "Prepared"),
    ("f", "If", "Qf", "f", "F"),
)


def _kde_density_region(i_values, q_values, probability=KDE_PROBABILITY, grid_size=KDE_GRID_SIZE):
    """Return a fast binned-KDE grid and highest-density region."""
    points = np.vstack((np.asarray(i_values, dtype=float), np.asarray(q_values, dtype=float)))
    finite = np.isfinite(points).all(axis=0)
    points = points[:, finite]
    if points.shape[1] < 3:
        return None
    if np.linalg.matrix_rank(np.cov(points)) < 2:
        return None

    padding = 0.15 * np.maximum(np.ptp(points, axis=1), np.std(points, axis=1))
    padding = np.maximum(padding, np.finfo(float).eps)
    i_edges = np.linspace(points[0].min() - padding[0], points[0].max() + padding[0], grid_size + 1)
    q_edges = np.linspace(points[1].min() - padding[1], points[1].max() + padding[1], grid_size + 1)
    i_axis = 0.5 * (i_edges[:-1] + i_edges[1:])
    q_axis = 0.5 * (q_edges[:-1] + q_edges[1:])
    i_grid, q_grid = np.meshgrid(i_axis, q_axis)

    # Histogram first, then apply Scott-bandwidth Gaussian smoothing. This
    # approximates gaussian_kde without its O(num_samples * grid_points) cost.
    histogram, _, _ = np.histogram2d(points[1], points[0], bins=(q_edges, i_edges))
    scott_factor = points.shape[1] ** (-1 / 6)
    grid_step = np.array((q_axis[1] - q_axis[0], i_axis[1] - i_axis[0]))
    bandwidth = scott_factor * np.array((np.std(points[1]), np.std(points[0])))
    sigma_in_bins = np.clip(bandwidth / grid_step, 0.75, grid_size / 4)
    density = gaussian_filter(histogram, sigma=sigma_in_bins, mode="nearest")

    interpolator = RegularGridInterpolator(
        (q_axis, i_axis),
        density,
        bounds_error=False,
        fill_value=0,
    )
    sample_density = interpolator(np.column_stack((points[1], points[0])))
    level = float(np.quantile(sample_density, 1 - probability))
    enclosed_fraction = float(np.mean(sample_density >= level))
    return i_grid, q_grid, density, level, enclosed_fraction


def _empty_kde_region():
    nan_grid = np.full((KDE_GRID_SIZE, KDE_GRID_SIZE), np.nan)
    return nan_grid.copy(), nan_grid.copy(), nan_grid.copy(), np.nan, np.nan


def _add_kde_regions(ds_fit: xr.Dataset, qubits) -> xr.Dataset:
    """Add 95% KDE regions in the acquired IQ coordinates."""
    state_specs = [spec for spec in STATE_SPECS if spec[1] in ds_fit and spec[2] in ds_fit]
    results = []
    for qubit in qubits:
        selected = ds_fit.sel(qubit=qubit.name)
        qubit_results = []
        for _, i_name, q_name, _, _ in state_specs:
            region = _kde_density_region(selected[i_name], selected[q_name])
            qubit_results.append(region if region is not None else _empty_kde_region())
        results.append(qubit_results)

    coords = {
        "qubit": ds_fit.qubit.data,
        "kde_y": np.arange(KDE_GRID_SIZE),
        "kde_x": np.arange(KDE_GRID_SIZE),
    }
    for state_index, (_, _, _, state_name, _) in enumerate(state_specs):
        ds_fit[f"{state_name}_kde_I"] = xr.DataArray(
            np.stack([result[state_index][0] for result in results]), dims=("qubit", "kde_y", "kde_x"), coords=coords
        )
        ds_fit[f"{state_name}_kde_Q"] = xr.DataArray(
            np.stack([result[state_index][1] for result in results]), dims=("qubit", "kde_y", "kde_x"), coords=coords
        )
        ds_fit[f"{state_name}_kde_density"] = xr.DataArray(
            np.stack([result[state_index][2] for result in results]), dims=("qubit", "kde_y", "kde_x"), coords=coords
        )
        ds_fit[f"{state_name}_kde_95_level"] = xr.DataArray(
            [result[state_index][3] for result in results], dims="qubit", coords={"qubit": ds_fit.qubit.data}
        )
        ds_fit[f"{state_name}_kde_enclosed_fraction"] = xr.DataArray(
            [result[state_index][4] for result in results], dims="qubit", coords={"qubit": ds_fit.qubit.data}
        )
    return ds_fit


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    iw_angle: float
    ge_threshold: float
    rus_threshold: float
    readout_fidelity: float
    average_fidelity: float
    fidelity_matrix: list
    center_separation: float
    separation_to_width: float
    confusion_matrix: list
    state_labels: list
    center_matrix: list | None
    threshold_pairs: list
    threshold_line_midpoints: list | None
    threshold_line_normals: list | None
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s = f"IW angle: {fit_results[q]['iw_angle'] * 180 / np.pi:.1f} deg | "
        s += f"ge_threshold: {fit_results[q]['ge_threshold'] * 1e3:.1f} mV | "
        s += f"rus_threshold: {fit_results[q]['rus_threshold'] * 1e3:.1f} mV | "
        s += f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} % | "
        s += f"separation/width: {fit_results[q]['separation_to_width']:.2f} \n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s)


def save_fit_results(run_directory, fit_results: Dict, filename: str = "fit_results.json") -> Path:
    """Save IQ-blobs fit results, including average fidelity and fidelity matrix."""
    output_path = Path(run_directory) / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(fit_results, file, indent=2)
        file.write("\n")
    return output_path


def log_blob_diagnostics(ds: xr.Dataset, log_callable=None):
    """Log raw cloud centers and widths to expose acquisition asymmetries."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qubit in ds.qubit.values:
        selected = ds.sel(qubit=qubit)
        ground_width = float(np.sqrt(selected.Ig.var() + selected.Qg.var()))
        prepared_width = float(np.sqrt(selected.Ie.var() + selected.Qe.var()))
        width_ratio = prepared_width / ground_width if ground_width else np.nan
        center_separation = float(
            np.hypot(selected.Ie.mean() - selected.Ig.mean(), selected.Qe.mean() - selected.Qg.mean())
        )
        pooled_width = np.sqrt((ground_width**2 + prepared_width**2) / 2)
        separation_to_width = center_separation / pooled_width if pooled_width else np.nan
        prepared_points = np.column_stack((selected.Ie.values, selected.Qe.values))
        prepared_unique_fraction = len(np.unique(prepared_points, axis=0)) / len(prepared_points)
        warnings = []
        if not 0.5 <= width_ratio <= 2:
            warnings.append("strongly asymmetric blob widths")
        if separation_to_width < 1:
            warnings.append("blob separation is smaller than measurement noise")
        warning = f" | WARNING: {', '.join(warnings)}" if warnings else ""
        log_callable(
            f"Blob diagnostics for {qubit}: ground width={ground_width * 1e3:.3f} mV | "
            f"prepared width={prepared_width * 1e3:.3f} mV | "
            f"width ratio={width_ratio:.3f} | center separation={center_separation * 1e3:.3f} mV | "
            f"separation/width={separation_to_width:.3f} | "
            f"prepared unique fraction={prepared_unique_fraction:.3f}{warning}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )
    iq_list = ["Ig", "Qg", "Ie", "Qe"]
    if "If" in ds and "Qf" in ds:
        iq_list.extend(["If", "Qf"])
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=iq_list)
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
    # Rotate the axis connecting the two blob means onto +I, independently for
    # each qubit. This also guarantees that the excited-state mean is above the
    # discrimination threshold.
    delta_i = ds_fit.Ie.mean(dim="n_runs") - ds_fit.Ig.mean(dim="n_runs")
    delta_q = ds_fit.Qe.mean(dim="n_runs") - ds_fit.Qg.mean(dim="n_runs")
    angle = np.arctan2(-delta_q, delta_i)
    C = np.cos(angle)
    S = np.sin(angle)
    ds_fit = ds_fit.assign({"iw_angle": xr.DataArray(angle, coords=dict(qubit=ds_fit.qubit.data))})

    ds_fit = ds_fit.assign({"Ig_rot": ds_fit.Ig * C - ds_fit.Qg * S})
    ds_fit = ds_fit.assign({"Qg_rot": ds_fit.Ig * S + ds_fit.Qg * C})
    ds_fit = ds_fit.assign({"Ie_rot": ds_fit.Ie * C - ds_fit.Qe * S})
    ds_fit = ds_fit.assign({"Qe_rot": ds_fit.Ie * S + ds_fit.Qe * C})
    if "If" in ds_fit and "Qf" in ds_fit:
        ds_fit = ds_fit.assign({"If_rot": ds_fit.If * C - ds_fit.Qf * S})
        ds_fit = ds_fit.assign({"Qf_rot": ds_fit.If * S + ds_fit.Qf * C})
    ds_fit = _add_kde_regions(ds_fit, node.namespace["qubits"])

    threshold = []
    gg, ge, eg, ee = [], [], [], []
    for q in node.namespace["qubits"]:
        ground = np.asarray(ds_fit.Ig_rot.sel(qubit=q.name), dtype=float)
        prepared = np.asarray(ds_fit.Ie_rot.sel(qubit=q.name), dtype=float)
        fitted_threshold = _optimal_threshold(ground, prepared)
        threshold.append(fitted_threshold)
        gg.append(np.mean(ground < fitted_threshold))
        ge.append(np.mean(ground > fitted_threshold))
        eg.append(np.mean(prepared < fitted_threshold))
        ee.append(np.mean(prepared > fitted_threshold))
    ds_fit = ds_fit.assign({"ge_threshold": xr.DataArray(threshold, coords=dict(qubit=ds_fit.qubit.data))})
    # Active-reset exit and standard state-discrimination use the same threshold.
    ds_fit = ds_fit.assign({"rus_threshold": ds_fit.ge_threshold.copy()})
    ds_fit = ds_fit.assign({"gg": xr.DataArray(gg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ge": xr.DataArray(ge, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"eg": xr.DataArray(eg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ee": xr.DataArray(ee, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign(
        {"readout_fidelity": xr.DataArray(100 * (ds_fit.gg + ds_fit.ee) / 2, coords=dict(qubit=ds_fit.qubit.data))}
    )
    ds_fit = ds_fit.assign({"average_fidelity": ds_fit.readout_fidelity.copy()})
    fidelity_matrices = np.stack(
        [
            np.asarray([[gg_value, ge_value], [eg_value, ee_value]], dtype=float)
            for gg_value, ge_value, eg_value, ee_value in zip(gg, ge, eg, ee)
        ]
    )
    ds_fit = ds_fit.assign(
        {
            "fidelity_matrix": xr.DataArray(
                fidelity_matrices,
                dims=("qubit", "fidelity_prepared_state", "fidelity_measured_state"),
                coords={
                    "qubit": ds_fit.qubit.data,
                    "fidelity_prepared_state": ["g", "e"],
                    "fidelity_measured_state": ["g", "e"],
                },
            )
        }
    )
    center_separation = np.hypot(
        ds_fit.Ie.mean(dim="n_runs") - ds_fit.Ig.mean(dim="n_runs"),
        ds_fit.Qe.mean(dim="n_runs") - ds_fit.Qg.mean(dim="n_runs"),
    )
    ground_width = np.sqrt(ds_fit.Ig.var(dim="n_runs") + ds_fit.Qg.var(dim="n_runs"))
    prepared_width = np.sqrt(ds_fit.Ie.var(dim="n_runs") + ds_fit.Qe.var(dim="n_runs"))
    pooled_width = np.sqrt((ground_width**2 + prepared_width**2) / 2)
    ds_fit = ds_fit.assign({"center_separation": center_separation})
    ds_fit = ds_fit.assign({"separation_to_width": center_separation / pooled_width})
    ds_fit = _add_state_centers_and_confusion(ds_fit)

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _optimal_threshold(ground, prepared):
    """Return the exact threshold that minimizes classification errors."""
    ground = np.asarray(ground, dtype=float)
    prepared = np.asarray(prepared, dtype=float)
    ground = ground[np.isfinite(ground)]
    prepared = prepared[np.isfinite(prepared)]
    if ground.size == 0 or prepared.size == 0:
        return np.nan

    values = np.concatenate((ground, prepared))
    labels = np.concatenate(
        (np.zeros(ground.size, dtype=np.int8), np.ones(prepared.size, dtype=np.int8))
    )
    unique_values, inverse = np.unique(values, return_inverse=True)
    ground_counts = np.bincount(inverse[labels == 0], minlength=unique_values.size)
    prepared_counts = np.bincount(inverse[labels == 1], minlength=unique_values.size)

    cumulative_ground = np.cumsum(ground_counts)
    cumulative_prepared = np.cumsum(prepared_counts)
    errors = np.concatenate(
        (
            [ground.size],
            ground.size - cumulative_ground + cumulative_prepared,
        )
    )
    best_index = int(np.argmin(errors))
    if best_index == 0:
        return float(np.nextafter(unique_values[0], -np.inf))
    if best_index == unique_values.size:
        return float(np.nextafter(unique_values[-1], np.inf))
    return float(0.5 * (unique_values[best_index - 1] + unique_values[best_index]))


def _add_state_centers_and_confusion(fit: xr.Dataset) -> xr.Dataset:
    """Add per-state IQ centers, nearest-center confusion matrices, and pairwise thresholds."""
    state_specs = [spec for spec in STATE_SPECS if spec[1] in fit and spec[2] in fit]
    state_labels = [spec[0] for spec in state_specs]
    threshold_pairs = _threshold_pairs_for_states(state_labels)
    centers_by_qubit = []
    confusion_by_qubit = []
    threshold_midpoints_by_qubit = []
    threshold_normals_by_qubit = []

    for q in fit.qubit.values:
        selected = fit.sel(qubit=q)
        centers = np.asarray(
            [
                [float(selected[i_name].mean()), float(selected[q_name].mean())]
                for _, i_name, q_name, _, _ in state_specs
            ]
        )
        confusion = np.zeros((len(state_specs), len(state_specs)))
        for prepared_index, (_, i_name, q_name, _, _) in enumerate(state_specs):
            points = np.column_stack((selected[i_name].values, selected[q_name].values))
            distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            measured = np.argmin(distances, axis=1)
            for measured_index in range(len(state_specs)):
                confusion[prepared_index, measured_index] = float(np.mean(measured == measured_index))
        centers_by_qubit.append(centers)
        confusion_by_qubit.append(confusion)
        midpoints = []
        normals = []
        for left_state, right_state in threshold_pairs:
            left_center = centers[state_labels.index(left_state)]
            right_center = centers[state_labels.index(right_state)]
            midpoints.append(0.5 * (left_center + right_center))
            normals.append(right_center - left_center)
        threshold_midpoints_by_qubit.append(np.asarray(midpoints))
        threshold_normals_by_qubit.append(np.asarray(normals))

    fit = fit.assign(
        {
            "state_center_matrix": xr.DataArray(
                np.stack(centers_by_qubit),
                dims=("qubit", "state", "IQ"),
                coords={"qubit": fit.qubit.data, "state": state_labels, "IQ": ["I", "Q"]},
            ),
            "state_confusion_matrix": xr.DataArray(
                np.stack(confusion_by_qubit),
                dims=("qubit", "prepared_state", "measured_state"),
                coords={
                    "qubit": fit.qubit.data,
                    "prepared_state": state_labels,
                    "measured_state": state_labels,
                },
            ),
            "threshold_line_midpoint": xr.DataArray(
                np.stack(threshold_midpoints_by_qubit),
                dims=("qubit", "threshold", "IQ"),
                coords={"qubit": fit.qubit.data, "threshold": threshold_pairs, "IQ": ["I", "Q"]},
            ),
            "threshold_line_normal": xr.DataArray(
                np.stack(threshold_normals_by_qubit),
                dims=("qubit", "threshold", "IQ"),
                coords={"qubit": fit.qubit.data, "threshold": threshold_pairs, "IQ": ["I", "Q"]},
            ),
        }
    )
    return fit


def _threshold_pairs_for_states(state_labels):
    """Return stable discriminator-pair names for acquired states."""
    if state_labels == ["g", "e", "f"]:
        return ["ge", "ef", "gf"]
    return [f"{state_labels[i]}{state_labels[j]}" for i in range(len(state_labels)) for j in range(i + 1, len(state_labels))]


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Assess whether the fit was successful or not
    nan_success = (
        np.isnan(fit.iw_angle)
        | np.isnan(fit.ge_threshold)
        | np.isnan(fit.rus_threshold)
        | np.isnan(fit.readout_fidelity)
    )
    success_criteria = ~nan_success & (fit.separation_to_width >= 1) & (fit.readout_fidelity >= 60)
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            iw_angle=float(fit.sel(qubit=q).iw_angle),
            ge_threshold=float(fit.sel(qubit=q).ge_threshold),
            rus_threshold=float(fit.sel(qubit=q).rus_threshold),
            readout_fidelity=float(fit.sel(qubit=q).readout_fidelity),
            average_fidelity=float(fit.sel(qubit=q).average_fidelity),
            fidelity_matrix=fit.sel(qubit=q).fidelity_matrix.values.tolist(),
            center_separation=float(fit.sel(qubit=q).center_separation),
            separation_to_width=float(fit.sel(qubit=q).separation_to_width),
            confusion_matrix=fit.sel(qubit=q).state_confusion_matrix.values.tolist(),
            state_labels=list(fit.state.values),
            center_matrix=fit.sel(qubit=q).state_center_matrix.values.tolist(),
            threshold_pairs=list(fit.threshold.values),
            threshold_line_midpoints=fit.sel(qubit=q).threshold_line_midpoint.values.tolist(),
            threshold_line_normals=fit.sel(qubit=q).threshold_line_normal.values.tolist(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
