from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    use_state_discrimination: bool = False,
):
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with(
            ax,
            ds,
            qubit,
            fits.sel(qubit=qubit["qubit"]),
            use_state_discrimination=use_state_discrimination,
        )

    grid.fig.suptitle("Rabi chevron")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    use_state_discrimination: bool = False,
):
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

    data = "state" if use_state_discrimination else "I"
    if data not in ds:
        raise RuntimeError(
            f"Rabi-chevron plot expected {data!r} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )
    scale = 1 if data == "state" else 1 / u.mV

    # Create a first x-axis for full_freq_GHz
    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz)[data] * scale).plot(
        ax=ax, y="pulse_duration", x="full_freq_GHz", add_colorbar=False
    )
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Pulse duration [ns]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz)[data] * scale).plot(
        ax=ax2, y="pulse_duration", x="detuning_MHz", add_colorbar=False
    )
    ax2.set_xlabel("Detuning [MHz]")
