from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit]
        selected.ground_phase.plot(ax=ax1, x="full_freq_GHz", label="Ground")
        selected.mixed_phase.plot(ax=ax1, x="full_freq_GHz", label="Mixed (saturation)")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        ax1.legend()
    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state phase")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state amplitude")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    if fit:
        fitted_data = lorentzian_dip(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit]
    (selected.ground_IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz", label="Ground")
    (selected.mixed_IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz", label="Mixed (saturation)")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    if fitted_data is not None:
        ax.plot(ds.full_freq.loc[qubit] / u.GHz, fitted_data / u.mV, "k--", label="Ground fit")
    ax.legend()


def plot_iq_response(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot ground and mixed-state resonator trajectories in the IQ plane."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        selected = ds.loc[qubit]
        ax.plot(selected.Ig / u.mV, selected.Qg / u.mV, ".-", label="Ground", markersize=3)
        ax.plot(selected.Im / u.mV, selected.Qm / u.mV, ".-", label="Mixed (saturation)", markersize=3)
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.axis("equal")
        ax.legend()

    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state IQ response")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
