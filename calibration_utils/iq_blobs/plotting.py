from typing import Any, List
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def _available_state_specs(ds: xr.Dataset):
    return [spec for spec in STATE_PLOT_SPECS if spec[1] in ds and spec[2] in ds]


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
        hspace=0.35,
    )

    for row, qubit in enumerate(qubits):
        qubit_grid = outer_grid[row].subgridspec(
            2,
            2,
            height_ratios=[1.5, 1],
            width_ratios=[1.4, 1],
            hspace=0.35,
            wspace=0.25,
        )
        iq_ax = fig.add_subplot(qubit_grid[0, 0])
        matrix_ax = fig.add_subplot(qubit_grid[0, 1])
        histogram_ax = fig.add_subplot(qubit_grid[1, :])

        fit = fits.sel(qubit=qubit.name)
        qubit_ref = {"qubit": qubit.name}
        plot_individual_iq_blobs(iq_ax, ds, qubit_ref, fit)
        plot_individual_confusion_matrix(matrix_ax, ds, qubit_ref, fit)
        plot_individual_histograms(histogram_ax, ds, qubit_ref, fit)

        status = "PASS" if bool(fit.success.values) else "FAIL"
        iq_ax.set_title(
            f"{qubit.name}: acquired IQ clouds ({status})\n"
            f"separation/width={float(fit.separation_to_width.values):.2f}, "
            f"fitted rotation={np.degrees(float(fit.iw_angle.values)):.1f} deg"
        )
        matrix_ax.set_title(f"{qubit.name}: confusion matrix")
        histogram_ax.set_title(
            f"{qubit.name}: rotated-I histogram\n"
            f"fidelity={float(fit.readout_fidelity.values):.1f}%"
        )

    fig.suptitle("IQ blobs calibration")
    metadata_lines = _format_iq_blobs_run_metadata(qubits, run_metadata)
    if metadata_lines:
        add_calibration_parameter_box(fig, metadata_lines, gid="iq_blobs_parameters")
        calibration_plot = CalibrationPlot(fig)
        calibration_plot.add_timestamp()
        calibration_plot.tight_layout_for_parameters(len(metadata_lines), top=0.93)
    else:
        fig.subplots_adjust(top=0.93)
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

    reset_type = run_metadata.get("reset_type")
    parameter_summaries = []
    parameter_summaries.append(f"operation={operation_name}")
    if reset_type is not None:
        parameter_summaries.append(f"active reset={reset_type == 'active'}")
    if run_metadata.get("num_shots") is not None:
        parameter_summaries.append(f"num reps={run_metadata['num_shots']}")
    if run_metadata.get("pi_repetitions") is not None:
        parameter_summaries.append(f"pi reps={run_metadata['pi_repetitions']}")
    if run_metadata.get("states") is not None:
        parameter_summaries.append(f"states={','.join(str(state) for state in run_metadata['states'])}")
    if run_metadata.get("qubit_operation") is not None:
        parameter_summaries.append(f"prep operation={run_metadata['qubit_operation']}")

    return ["Parameters", *readout_summaries, " | ".join(parameter_summaries)]


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
    for _, i_name, q_name, _, _, label, color, _, alpha in _available_state_specs(raw):
        ax.plot(
            1e3 * raw[i_name],
            1e3 * raw[q_name],
            ".",
            alpha=alpha,
            label=label,
            markersize=2,
            color=color,
            zorder=1,
        )
        center = (float(raw[i_name].mean()) * 1e3, float(raw[q_name].mean()) * 1e3)
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
        ax.contour(
            1e3 * fit[f"{state_name}_kde_I"].values,
            1e3 * fit[f"{state_name}_kde_Q"].values,
            fit[f"{state_name}_kde_density"].values,
            levels=[float(fit[level_name].values)],
            colors=[contour_color],
            linewidths=1.5,
        )
        ax.plot([], [], color=contour_color, linewidth=1.5, label=f"{label} 95% KDE")

    ax.axis("equal")
    if "If" in raw and "Qf" in raw:
        _plot_pairwise_threshold_lines(ax, fit)
    else:
        _plot_raw_threshold(ax, fit.rus_threshold, fit.iw_angle, color="k", label="RUS Threshold")
        _plot_raw_threshold(ax, fit.ge_threshold, fit.iw_angle, color="r", label="Threshold")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(f"{qubit['qubit']}\nFitted rotation={np.degrees(float(fit.iw_angle)):.1f} deg")
    ax.legend(fontsize="small")


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
