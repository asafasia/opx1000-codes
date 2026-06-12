from typing import List
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_iq_blobs_dashboard(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot IQ clouds, rotated-I histograms, and confusion matrices together."""
    fig = plt.figure(figsize=(14, max(8, 8 * len(qubits))))
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
            f"{qubit.name}: IQ clouds ({status})\n"
            f"separation/width={float(fit.separation_to_width.values):.2f}"
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_iq_blobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg.legend_handles[0].set_markersize(6)
    leg.legend_handles[1].set_markersize(6)
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    grid.fig.set_size_inches(15, 9)
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

    ax.plot(1e3 * fit.Ig_rot, 1e3 * fit.Qg_rot, ".", alpha=0.5, label="Ground", markersize=2)
    ax.plot(
        1e3 * fit.Ie_rot,
        1e3 * fit.Qe_rot,
        ".",
        alpha=1,
        label="Prepared",
        markersize=2,
    )
    g_center = (float(fit.Ig_rot.mean()) * 1e3, float(fit.Qg_rot.mean()) * 1e3)
    e_center = (float(fit.Ie_rot.mean()) * 1e3, float(fit.Qe_rot.mean()) * 1e3)
    ax.axvline(
        1e3 * fit.rus_threshold,
        color="k",
        linestyle="--",
        lw=0.5,
        label="RUS Threshold",
    )
    ax.plot(*g_center, "ko", markersize=6, label="Ground center")
    ax.plot(*e_center, "ro", markersize=6, label="Prepared center")

    ax.axvline(1e3 * fit.ge_threshold, color="r", linestyle="--", lw=0.5, label="Threshold")
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend(fontsize="small")


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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_histograms(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle("g.s. and e.s. histograms (rotated)")
    grid.fig.set_size_inches(15, 9)
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
    ax.axvline(
        1e3 * fit.rus_threshold,
        color="k",
        linestyle="--",
        lw=0.5,
        label="RUS Threshold",
    )
    ax.axvline(1e3 * fit.ge_threshold, color="r", linestyle="--", lw=0.5, label="Threshold")
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_confusion_matrix(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("g.s. and e.s. fidelity")
    grid.fig.set_size_inches(15, 9)
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
