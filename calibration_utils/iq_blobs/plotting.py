from typing import Any, List
from .analysis import _cloud_center
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import (
    FIGURE_SIZE,
    CalibrationPlot,
    add_calibration_parameter_box,
    format_readout_parameter_lines,
    qubit_grid_locations,
)

u = unit(coerce_to_integer=True)

STATE_PLOT_SPECS = (
    ("g", "Ig", "Qg", "Ig_rot", "ground", "Ground", "tab:blue", "navy", 0.35),
    (
        "e",
        "Ie",
        "Qe",
        "Ie_rot",
        "prepared",
        "Prepared",
        "tab:orange",
        "darkred",
        0.35,
    ),
    ("f", "If", "Qf", "If_rot", "f", "F", "tab:green", "darkgreen", 0.35),
)

PI_PULSE_TYPES = {
    "SquarePulse": "constant",
    "DragGaussianPulse": "drag",
    "DragCosinePulse": "cosine",
}


def _available_state_specs(ds: xr.Dataset):
    return [spec for spec in STATE_PLOT_SPECS if spec[1] in ds and spec[2] in ds]


def _is_three_state_fit(fit: xr.Dataset) -> bool:
    """Return whether the fit contains the G/E/F pairwise discriminators."""
    if "threshold" not in fit.coords:
        return False
    return {str(pair) for pair in fit.threshold.values} == {"ge", "ef", "gf"}


def plot_iq_blobs_dashboard(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    run_metadata: dict[str, Any] | None = None,
) -> Figure:
    """Plot acquired IQ clouds, rotated-I histograms, and confusion matrices."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    outer_grid = fig.add_gridspec(
        len(qubits),
        1,
        hspace=0.9,
    )

    for row, qubit in enumerate(qubits):
        fit = fits.sel(qubit=qubit.name)
        qubit_ref = {"qubit": qubit.name}
        three_state = _is_three_state_fit(fit)
        if three_state:
            qubit_grid = outer_grid[row].subgridspec(
                2,
                6,
                height_ratios=[1.5, 1],
                hspace=0.9,
                wspace=0.35,
            )
            iq_ax = fig.add_subplot(qubit_grid[0, :4])
            matrix_ax = fig.add_subplot(qubit_grid[0, 4:])
            histogram_axes = [
                fig.add_subplot(qubit_grid[1, 0:2]),
                fig.add_subplot(qubit_grid[1, 2:4]),
                fig.add_subplot(qubit_grid[1, 4:6]),
            ]
        else:
            qubit_grid = outer_grid[row].subgridspec(
                2,
                2,
                height_ratios=[1.5, 1],
                width_ratios=[1.4, 1],
                hspace=0.9,
                wspace=0.25,
            )
            iq_ax = fig.add_subplot(qubit_grid[0, 0])
            matrix_ax = fig.add_subplot(qubit_grid[0, 1])
            histogram_axes = [fig.add_subplot(qubit_grid[1, :])]

        plot_individual_iq_blobs(iq_ax, ds, qubit_ref, fit)
        plot_individual_confusion_matrix(matrix_ax, ds, qubit_ref, fit)
        if three_state:
            for histogram_ax, pair in zip(histogram_axes, ("ge", "ef", "gf")):
                plot_pairwise_rotated_histogram(histogram_ax, ds, qubit_ref, fit, pair)
        else:
            plot_individual_histograms(histogram_axes[0], ds, qubit_ref, fit)

        status = "PASS" if bool(fit.success.values) else "FAIL"
        iq_view = "G-E aligned IQ clouds" if "If" in ds and "Qf" in ds else "acquired IQ clouds"
        iq_ax.set_title(
            f"{qubit.name}: {iq_view} ({status})\n"
            f"separation/width={float(fit.separation_to_width.values):.2f}, "
            f"fitted rotation={np.degrees(float(fit.iw_angle.values)):.1f} deg"
        )
        matrix_ax.set_title(f"{qubit.name}: confusion matrix")
        if not three_state:
            histogram_axes[0].set_title(
                f"{qubit.name}: rotated-I histogram\n"
                f"fidelity={float(fit.readout_fidelity.values):.1f}%"
            )

    fig.suptitle("IQ blobs calibration")
    metadata_lines = _format_iq_blobs_run_metadata(qubits, run_metadata)
    if metadata_lines:
        add_calibration_parameter_box(fig, metadata_lines, gid="iq_blobs_parameters")
        calibration_plot = CalibrationPlot(fig)
        calibration_plot.add_timestamp()
        calibration_plot.tight_layout_for_parameters(len(metadata_lines), top=0.88)
    else:
        fig.subplots_adjust(top=0.88)
    return fig


def _format_iq_blobs_run_metadata(
    qubits: List[AnyTransmon],
    run_metadata: dict[str, Any] | None,
) -> list[str]:
    """Return compact run-parameter lines for the dashboard parameter box."""
    if not run_metadata:
        return []

    operation_name = run_metadata.get("operation", "readout")
    readout_summaries = format_readout_parameter_lines(qubits, operation=operation_name)
    readout_mode_summaries = []
    for qubit in qubits:
        resonator = getattr(qubit, "resonator", None)
        use_kernel = (getattr(resonator, "readout_gef", {}).get("use_kernel")
                      if operation_name == "readout_GEF" else getattr(resonator, "use_kernel", None))
        xy_operations = getattr(getattr(qubit, "xy", None), "operations", {})
        pi_pulse = (
            xy_operations.get("x180") if hasattr(xy_operations, "get") else None
        )
        pi_pulse_type = (
            PI_PULSE_TYPES.get(type(pi_pulse).__name__, type(pi_pulse).__name__)
            if pi_pulse is not None
            else None
        )
        mode_parts = []
        if use_kernel is not None:
            mode_parts.append(f"optimized kernel={bool(use_kernel)}")
        if pi_pulse_type is not None:
            mode_parts.append(f"pi pulse type={pi_pulse_type}")
        if mode_parts:
            readout_mode_summaries.append(
                f"{qubit.name}: " + " | ".join(mode_parts)
            )

    reset_type = run_metadata.get("reset_type")
    parameter_summaries = []
    parameter_summaries.append(f"operation={operation_name}")
    discriminator = run_metadata.get("readout_discriminator")
    if discriminator is not None:
        parameter_summaries.append(f"discrimination={discriminator}")
    if reset_type is not None:
        parameter_summaries.append(
            f"active reset={reset_type in {'active', 'active_gef'}}"
        )
    if run_metadata.get("num_shots") is not None:
        parameter_summaries.append(f"num reps={run_metadata['num_shots']}")
    if run_metadata.get("pi_repetitions") is not None:
        parameter_summaries.append(f"pi reps={run_metadata['pi_repetitions']}")
    if run_metadata.get("states") is not None:
        parameter_summaries.append(f"states={','.join(str(state) for state in run_metadata['states'])}")
    if run_metadata.get("qubit_operation") is not None:
        parameter_summaries.append(f"prep operation={run_metadata['qubit_operation']}")

    return [
        "Parameters",
        *readout_summaries,
        *readout_mode_summaries,
        " | ".join(parameter_summaries),
    ]


def plot_iq_blobs(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        plot_individual_iq_blobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg.legend_handles[0].set_markersize(6)
    leg.legend_handles[1].set_markersize(6)
    grid.fig.suptitle("g.s. and e.s. discriminators (acquired IQ coordinates)")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def _robust_iq_limits(raw: xr.Dataset, sigma_extent: float = 4.0):
    """Frame the cloud cores using medians and robust per-quadrature sigma.

    MAD/IQR estimates avoid the outlier sensitivity of ordinary variance.
    Only the viewport changes; all shots and calibrated centers are retained.
    """
    centers, widths = [], []
    for spec in _available_state_specs(raw):
        cloud = 1e3 * np.column_stack((raw[spec[1]].values.ravel(), raw[spec[2]].values.ravel()))
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if not len(cloud):
            continue
        center = np.median(cloud, axis=0)
        mad_sigma = 1.4826 * np.median(np.abs(cloud - center), axis=0)
        quartiles = np.percentile(cloud, [25, 75], axis=0)
        iqr_sigma = (quartiles[1] - quartiles[0]) / 1.349
        centers.append(center)
        widths.append(np.maximum(mad_sigma, iqr_sigma))
    if not centers:
        return None
    centers, widths = np.asarray(centers), np.asarray(widths)
    # Keep zero-width/quantized clouds visible without falling back to extrema.
    scale = max(float(np.max(np.ptp(centers, axis=0))), float(np.max(widths)), 1e-3)
    half_width = np.maximum(sigma_extent * widths, 0.02 * scale)
    lower = np.min(centers - half_width, axis=0)
    upper = np.max(centers + half_width, axis=0)
    padding = 0.06 * (upper - lower)
    return tuple(zip(lower - padding, upper + padding))


def _ge_aligned_iq_view(raw: xr.Dataset, fit: xr.Dataset):
    """Rotate display copies so fitted G/E centers share Q; never change calibration."""
    raw_view, fit_view = raw.copy(deep=True), fit.copy(deep=True)
    specs = _available_state_specs(raw)
    if "state_center_matrix" not in fit_view:
        method = fit.attrs.get("center_method", "mean")
        fit_view["state_center_matrix"] = xr.DataArray(
            [[float(_cloud_center(raw[spec[1]], method)),
              float(_cloud_center(raw[spec[2]], method))] for spec in specs],
            dims=("state", "IQ"), coords={"state": [spec[0] for spec in specs], "IQ": ["I", "Q"]})
    centers = fit_view.state_center_matrix
    delta = (centers.sel(state="e") - centers.sel(state="g")).sel(IQ=["I", "Q"]).values
    angle = float(np.arctan2(-delta[1], delta[0]))
    if not np.isfinite(angle):
        return raw_view, fit_view, 0.
    c, s = np.cos(angle), np.sin(angle)
    for spec in specs:
        i_name, q_name, state_name = spec[1], spec[2], spec[4]
        raw_view[i_name] = raw[i_name] * c - raw[q_name] * s
        raw_view[q_name] = raw[i_name] * s + raw[q_name] * c
        ki, kq = f"{state_name}_kde_I", f"{state_name}_kde_Q"
        if ki in fit and kq in fit:
            fit_view[ki] = fit[ki] * c - fit[kq] * s
            fit_view[kq] = fit[ki] * s + fit[kq] * c
    for name in ("state_center_matrix", "threshold_line_midpoint", "threshold_line_normal"):
        if name in fit_view:
            values = fit_view[name]
            i, q = values.sel(IQ="I", drop=True), values.sel(IQ="Q", drop=True)
            fit_view[name] = xr.concat([i * c - q * s, i * s + q * c],
                dim=xr.IndexVariable("IQ", ["I", "Q"])).transpose(*values.dims)
    return raw_view, fit_view, angle


def plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    raw = ds.sel(qubit=qubit["qubit"])
    ge_aligned = all(name in raw for name in ("Ig", "Qg", "Ie", "Qe", "If", "Qf"))
    if ge_aligned:
        raw, fit, display_angle = _ge_aligned_iq_view(raw, fit)
    specs = _available_state_specs(raw)
    view_points = []  # Centers and contour vertices that must remain visible.
    # Render one mixed collection, rather than putting every F point on top of
    # all G/E points. A local fixed seed makes saved plots reproducible and does
    # not modify acquisition ordering or the global random state.
    points, colors = [], []
    for _, i_name, q_name, _, _, label, color, _, _ in specs:
        cloud = 1e3 * np.column_stack((raw[i_name].values.ravel(), raw[q_name].values.ravel()))
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        points.append(cloud)
        colors.append(np.tile(to_rgb(color), (len(cloud), 1)))
    if points and sum(len(cloud) for cloud in points):
        points = np.concatenate(points)
        colors = np.concatenate(colors)
        order = np.random.default_rng(0).permutation(len(points))
        cloud_artist = ax.scatter(
            points[order, 0], points[order, 1], c=colors[order],
            s=4, alpha=0.16, edgecolors="none", linewidths=0,
            rasterized=True, zorder=1,
        )
        cloud_artist.set_gid("iq_mixed_clouds")
    for state, i_name, q_name, _, _, label, color, _, _ in specs:
        # Readable legend swatches are independent of the faint data markers.
        ax.plot([], [], ".", color=color, markersize=6, label=label, zorder=1)
        if "state_center_matrix" in fit:
            center = 1e3 * fit.state_center_matrix.sel(state=state, IQ=["I", "Q"]).values
        else:
            method = fit.attrs.get("center_method", "mean")
            center = (float(_cloud_center(raw[i_name], method)) * 1e3,
                      float(_cloud_center(raw[q_name], method)) * 1e3)
        view_points.append(np.asarray(center, dtype=float).reshape(1, 2))
        ax.plot(
            *center,
            "o",
            color=color,
            markeredgecolor="black",
            markersize=6,
            label=f"{label} center",
            zorder=10,
        )
    for _, _, _, _, state_name, label, _, contour_color, _ in _available_state_specs(raw):
        level_name = f"{state_name}_kde_95_level"
        if level_name not in fit or not np.isfinite(float(fit[level_name].values)):
            continue
        contour = ax.contour(
            1e3 * fit[f"{state_name}_kde_I"].values,
            1e3 * fit[f"{state_name}_kde_Q"].values,
            fit[f"{state_name}_kde_density"].values,
            levels=[float(fit[level_name].values)],
            colors=[contour_color],
            linewidths=1.5,
        )
        view_points.extend(segment for level in contour.allsegs for segment in level if len(segment))
        ax.plot([], [], color=contour_color, linewidth=1.5, label=f"{label} 95% KDE")

    # Expand the shorter data span, never crop a cloud to enforce equal scale.
    # Keep the outlier-aware zoom, while including fitted centers and contours.
    limits = _robust_iq_limits(raw)
    if limits is not None:
        bounds = np.asarray(limits, dtype=float)
        if view_points:
            vertices = np.concatenate(view_points)
            vertices = vertices[np.isfinite(vertices).all(axis=1)]
            if len(vertices):
                bounds[:, 0] = np.minimum(bounds[:, 0], vertices.min(axis=0))
                bounds[:, 1] = np.maximum(bounds[:, 1], vertices.max(axis=0))
        midpoint = bounds.mean(axis=1)
        half_span = 0.53 * np.max(bounds[:, 1] - bounds[:, 0])
        ax.set_xlim(midpoint[0] - half_span, midpoint[0] + half_span)
        ax.set_ylim(midpoint[1] - half_span, midpoint[1] + half_span)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    if "If" in raw and "Qf" in raw:
        if _is_three_state_fit(fit):
            _plot_nearest_center_boundary(ax, fit, raw)
        else:
            _plot_pairwise_threshold_lines(ax, fit)
    else:
        _plot_raw_threshold(ax, fit.rus_threshold, fit.iw_angle, color="k", label="RUS Threshold")
        _plot_raw_threshold(ax, fit.ge_threshold, fit.iw_angle, color="r", label="Threshold")
    ax.set_xlabel("I rotated [mV]" if ge_aligned else "I [mV]")
    ax.set_ylabel("Q rotated [mV]" if ge_aligned else "Q [mV]")
    rotation = display_angle if ge_aligned else float(fit.iw_angle)
    ax.set_title(f"{qubit['qubit']}\n{'G-E display' if ge_aligned else 'Fitted'} rotation={np.degrees(rotation):.1f} deg")
    ax.legend(
        fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, borderaxespad=0, columnspacing=1.2,
        handletextpad=0.5,
    ).set_zorder(20)


def _plot_raw_threshold(ax: Axes, threshold, angle, color: str, label: str):
    """Draw an I-rotated threshold in the acquired IQ coordinate system."""
    threshold_mv = 1e3 * float(threshold)
    angle = float(angle)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    i_limits = ax.get_xlim()
    q_limits = ax.get_ylim()

    if abs(sine) < 1e-12:
        ax.axvline(threshold_mv / cosine, color=color, linestyle="--", lw=0.5, label=label)
    else:
        i_values = np.asarray(i_limits)
        q_values = (cosine * i_values - threshold_mv) / sine
        ax.plot(i_values, q_values, color=color, linestyle="--", lw=0.5, label=label)
    ax.set_xlim(i_limits)
    ax.set_ylim(q_limits)


def _nearest_center_boundary_segments(centers, i_limits, q_limits):
    """Clip pair bisectors to where that pair is jointly closest, and to the view.

    For three noncollinear centers this is the Voronoi Y junction, not three
    intersecting infinite lines. Collinear centers correctly give parallel
    separators. Work in normalized plot coordinates to keep tolerances stable.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        return []
    limits = np.array([sorted(i_limits), sorted(q_limits)], dtype=float)
    origin = limits.mean(axis=1)
    scale = float(np.max(np.ptp(limits, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return []
    centers = np.unique((centers - origin) / scale, axis=0)
    limits = (limits - origin[:, None]) / scale
    segments = []
    for i, left in enumerate(centers):
        for j in range(i + 1, len(centers)):
            right = centers[j]
            normal = right - left
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            midpoint = (left + right) / 2
            direction = np.array([-normal[1], normal[0]]) / norm
            lower, upper = -np.inf, np.inf
            # Constraints n dot (midpoint + t*direction) <= bound.
            constraints = [
                (np.array([1., 0.]), limits[0, 1]),
                (np.array([-1., 0.]), -limits[0, 0]),
                (np.array([0., 1.]), limits[1, 1]),
                (np.array([0., -1.]), -limits[1, 0]),
            ]
            for k, other in enumerate(centers):
                if k not in (i, j):
                    n = other - left
                    constraints.append((n, np.dot(n, (left + other) / 2)))
            for n, bound in constraints:
                slope = float(np.dot(n, direction))
                remaining = float(bound - np.dot(n, midpoint))
                if abs(slope) < 1e-12:
                    if remaining < -1e-12:
                        lower, upper = 1., 0.
                        break
                elif slope > 0:
                    upper = min(upper, remaining / slope)
                else:
                    lower = max(lower, remaining / slope)
            if np.isfinite([lower, upper]).all() and upper - lower > 1e-12:
                segments.append(origin + scale * (midpoint + np.array([lower, upper])[:, None] * direction))
    return segments


def _nearest_center_regions(centers, i_limits, q_limits):
    """Return each center's Voronoi cell clipped to the visible rectangle."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        return []
    limits = np.array([sorted(i_limits), sorted(q_limits)], dtype=float)
    origin = limits.mean(axis=1)
    scale = float(np.max(np.ptp(limits, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return []
    normalized = (centers - origin) / scale
    bounds = (limits - origin[:, None]) / scale
    rectangle = np.array([[bounds[0, 0], bounds[1, 0]],
                          [bounds[0, 1], bounds[1, 0]],
                          [bounds[0, 1], bounds[1, 1]],
                          [bounds[0, 0], bounds[1, 1]]])
    regions = []
    for i, center in enumerate(normalized):
        polygon = rectangle.copy()
        for j, other in enumerate(normalized):
            if i == j:
                continue
            normal = other - center
            if np.linalg.norm(normal) < 1e-12:
                if j < i:
                    polygon = np.empty((0, 2))
                    break
                continue
            midpoint = (center + other) / 2
            clipped = []
            for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                start_distance = float(np.dot(start - midpoint, normal))
                end_distance = float(np.dot(end - midpoint, normal))
                start_inside, end_inside = start_distance <= 0, end_distance <= 0
                if start_inside != end_inside:
                    fraction = start_distance / (start_distance - end_distance)
                    clipped.append(start + fraction * (end - start))
                if end_inside:
                    clipped.append(end)
            polygon = np.asarray(clipped).reshape(-1, 2)
            if not len(polygon):
                break
        regions.append(origin + scale * polygon)
    return regions


def _plot_nearest_center_boundary(ax: Axes, fit: xr.Dataset, raw: xr.Dataset):
    """Draw the G/E/F threshold as one high-contrast decision-boundary object."""
    specs = _available_state_specs(raw)
    if "state_center_matrix" in fit:
        centers = np.asarray(fit.state_center_matrix.sel(state=[spec[0] for spec in specs], IQ=["I", "Q"]).values, dtype=float)
    else:
        centers = np.array([[float(_cloud_center(raw[spec[1]], fit.attrs.get("center_method", "mean"))), float(_cloud_center(raw[spec[2]], fit.attrs.get("center_method", "mean")))]
                            for spec in _available_state_specs(raw)])
    i_limits, q_limits = ax.get_xlim(), ax.get_ylim()
    segments = _nearest_center_boundary_segments(1e3 * centers, i_limits, q_limits)
    if not segments:
        return
    regions = _nearest_center_regions(1e3 * centers, i_limits, q_limits)
    for spec, region in zip(specs, regions):
        if len(region) < 3:
            continue
        patch = Polygon(region, closed=True, facecolor=spec[6], alpha=0.07,
                        edgecolor="none", linewidth=0, zorder=0)
        patch.set_gid(f"iq_decision_region_{spec[0]}")
        ax.add_patch(patch)
    boundary = LineCollection(segments, colors="0.15", linewidths=1.8,
                              label="G/E/F decision boundary", zorder=8)
    boundary.set_gid("iq_decision_boundary")
    ax.add_collection(boundary, autolim=False)
    # Mark a visible three-way junction, without inventing one for collinear
    # centers or bringing an off-screen vertex into the measured cloud range.
    endpoints = np.asarray(segments).reshape(-1, 2)
    tolerance = 1e-8 * max(np.ptp(i_limits), np.ptp(q_limits))
    for point in endpoints:
        if np.count_nonzero(np.linalg.norm(endpoints - point, axis=1) <= tolerance) >= 3:
            ax.plot(*point, "o", color="0.15", markeredgecolor="0.15",
                    markersize=5, zorder=9, label="_nolegend_")
            break
    ax.set_xlim(i_limits)
    ax.set_ylim(q_limits)


def _plot_pairwise_threshold_lines(ax: Axes, fit: xr.Dataset):
    """Draw pairwise center-bisector discriminator lines in acquired IQ coordinates."""
    if "threshold_line_midpoint" not in fit or "threshold_line_normal" not in fit:
        return
    i_limits = np.asarray(ax.get_xlim(), dtype=float)
    q_limits = np.asarray(ax.get_ylim(), dtype=float)
    span = max(np.ptp(i_limits), np.ptp(q_limits))
    if not np.isfinite(span) or span == 0:
        return

    colors = {"ge": "r", "ef": "purple", "gf": "k"}
    for pair in fit.threshold.values:
        pair_name = str(pair)
        midpoint = 1e3 * np.asarray(fit.threshold_line_midpoint.sel(threshold=pair).values, dtype=float)
        normal = np.asarray(fit.threshold_line_normal.sel(threshold=pair).values, dtype=float)
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm == 0:
            continue
        direction = np.asarray([-normal[1], normal[0]], dtype=float) / norm
        points = np.vstack((midpoint - span * direction, midpoint + span * direction))
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=colors.get(pair_name, "0.25"),
            linestyle="--",
            lw=0.8,
            label=f"{pair_name.upper()} threshold",
        )
    ax.set_xlim(i_limits)
    ax.set_ylim(q_limits)


def plot_historams(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        plot_individual_histograms(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle("g.s. and e.s. histograms (rotated)")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_histograms(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    for _, _, _, rot_name, _, label, color, _, alpha in STATE_PLOT_SPECS:
        if rot_name in fit:
            ax.hist(1e3 * fit[rot_name], bins=100, alpha=alpha, label=label, color=color)
    i_limits = ax.get_xlim()
    ax.axvline(
        1e3 * fit.rus_threshold,
        color="k",
        linestyle="--",
        lw=0.5,
        label="RUS Threshold",
    )
    ax.axvline(1e3 * fit.ge_threshold, color="r", linestyle="--", lw=0.5, label="Threshold")
    ax.set_xlim(i_limits)
    ax.set_xlabel("I Rotated [mV]")
    ax.set_ylabel("Counts")
    ax.set_title(qubit["qubit"])
    ax.legend(fontsize="small")


def plot_pairwise_rotated_histogram(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset,
    pair: str,
):
    """Plot two state clouds projected onto their pairwise center-separation axis."""
    raw = ds.sel(qubit=qubit["qubit"])
    specs = {spec[0]: spec for spec in _available_state_specs(raw)}
    left_state, right_state = pair
    if left_state not in specs or right_state not in specs:
        raise ValueError(f"Cannot plot {pair.upper()} histogram without both state clouds.")

    normal = np.asarray(
        fit.threshold_line_normal.sel(threshold=pair).values,
        dtype=float,
    )
    normal_norm = np.linalg.norm(normal)
    if not np.isfinite(normal_norm) or normal_norm == 0:
        raise ValueError(f"Cannot rotate {pair.upper()} histogram along a zero-length axis.")
    projection_axis = normal / normal_norm

    for state in (left_state, right_state):
        _, i_name, q_name, _, _, label, color, _, alpha = specs[state]
        projection = projection_axis[0] * raw[i_name] + projection_axis[1] * raw[q_name]
        ax.hist(1e3 * projection, bins=100, alpha=alpha, label=label, color=color)

    midpoint = np.asarray(
        fit.threshold_line_midpoint.sel(threshold=pair).values,
        dtype=float,
    )
    threshold = float(np.dot(midpoint, projection_axis))
    x_limits = ax.get_xlim()
    ax.axvline(
        1e3 * threshold,
        color={"ge": "r", "ef": "purple", "gf": "k"}.get(pair, "0.25"),
        linestyle="--",
        linewidth=0.8,
        label=f"{pair.upper()} threshold",
    )
    ax.set_xlim(x_limits)
    angle_degrees = np.degrees(np.arctan2(projection_axis[1], projection_axis[0]))
    ax.set_xlabel("Pairwise rotated I [mV]")
    ax.set_ylabel("Counts")
    ax.set_title(f"{pair.upper()} projection ({angle_degrees:.1f}°)")
    ax.legend(fontsize="small")


def plot_confusion_matrices(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the confusion matrix for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        plot_individual_confusion_matrix(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("g.s. and e.s. fidelity")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_confusion_matrix(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    if "state_confusion_matrix" in fit:
        confusion = np.asarray(fit.state_confusion_matrix.values, dtype=float)
        state_labels = [str(value) for value in fit.prepared_state.values]
    else:
        confusion = np.array([[float(fit.gg), float(fit.ge)], [float(fit.eg), float(fit.ee)]])
        state_labels = ["g", "e"]
    ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
    ticks = np.arange(len(state_labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels=[f"|{state}>" for state in state_labels])
    ax.set_yticklabels(labels=[f"|{state}>" for state in state_labels])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    for prepared in range(len(state_labels)):
        for measured in range(len(state_labels)):
            value = confusion[prepared, measured]
            ax.text(
                measured,
                prepared,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
    ax.set_title(qubit["qubit"])
