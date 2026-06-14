from typing import List
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

u = unit(coerce_to_integer=True)


def plot_iq_blobs_dashboard(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
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
    fig.subplots_adjust(top=0.93)
    return fig


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
    ax.plot(1e3 * raw.Ig, 1e3 * raw.Qg, ".", alpha=0.8, label="Ground", markersize=2)
    ax.plot(
        1e3 * raw.Ie,
        1e3 * raw.Qe,
        ".",
        alpha=0.5,
        label="Prepared",
        markersize=2,
    )
    g_center = (float(raw.Ig.mean()) * 1e3, float(raw.Qg.mean()) * 1e3)
    e_center = (float(raw.Ie.mean()) * 1e3, float(raw.Qe.mean()) * 1e3)
    ax.plot(*g_center, "ko", markersize=6, label="Ground center")
    ax.plot(*e_center, "ro", markersize=6, label="Prepared center")
    for state_name, color, label in (
        ("ground", "navy", "Ground 95% KDE"),
        ("prepared", "darkred", "Prepared 95% KDE"),
    ):
        level_name = f"{state_name}_kde_95_level"
        if level_name not in fit or not np.isfinite(float(fit[level_name].values)):
            continue
        ax.contour(
            1e3 * fit[f"{state_name}_kde_I"].values,
            1e3 * fit[f"{state_name}_kde_Q"].values,
            fit[f"{state_name}_kde_density"].values,
            levels=[float(fit[level_name].values)],
            colors=[color],
            linewidths=1.5,
        )
        ax.plot([], [], color=color, linewidth=1.5, label=label)

    ax.axis("equal")
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

    ax.hist(1e3 * fit.Ig_rot, bins=100, alpha=0.5, label="Ground")
    ax.hist(1e3 * fit.Ie_rot, bins=100, alpha=0.5, label="Prepared")
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

    confusion = np.array([[float(fit.gg), float(fit.ge)], [float(fit.eg), float(fit.ee)]])
    ax.imshow(confusion, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels=["|g>", "|e>"])
    ax.set_yticklabels(labels=["|g>", "|e>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    for prepared in range(2):
        for measured in range(2):
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
