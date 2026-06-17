from typing import List
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

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
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Single qubit randomized benchmarking")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    fit_succeeded = "success" in fit and bool(fit.success.values)
    if hasattr(fit, "population"):
        data = fit.population
        label = "Ground-state population"
    elif hasattr(fit, "state"):
        data = 1 - fit.state
        label = "Ground-state population"
    elif hasattr(fit, "I"):
        data = fit.I
        label = "I quadrature [mV]"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    data_std = data.std(dim="nb_of_sequences") / np.sqrt(ds.nb_of_sequences.size)
    ax.errorbar(
        fit.depths,
        fit.averaged_data,
        yerr=data_std,
        fmt=".",
        markersize=10,
        capsize=2,
        elinewidth=0.5,
    )
    ax.grid("all")
    ax.set_title(qubit["qubit"], pad=22)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel(label)
    if fit_succeeded:
        fitted = decay_exp(
            fit.depths,
            fit.fit_data.sel(fit_vals="a"),
            fit.fit_data.sel(fit_vals="offset"),
            fit.fit_data.sel(fit_vals="decay"),
        )
        ax.plot(fit.depths, fitted, "r--", label="fit")
        fit_text = f"1Q RB fidelity = {100 * float(fit.fidelity.values):.3f}"
        if "fidelity_std" in fit and not np.isnan(float(fit.fidelity_std.values)):
            fit_text += f" +/- {100 * float(fit.fidelity_std.values):.3f}"
        fit_text += "%"
    else:
        fit_text = "RB decay fit failed"
    ax.text(0.15, 0.9, fit_text, transform=ax.transAxes)
