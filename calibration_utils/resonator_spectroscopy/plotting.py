from typing import List
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

u = unit(coerce_to_integer=True)


def _add_detuning_axis(ax, current_frequency_ghz: float):
    """Add a top x-axis showing detuning from the configured resonance."""
    detuning_axis = ax.secondary_xaxis(
        "top",
        functions=(
            lambda frequency_ghz: (frequency_ghz - current_frequency_ghz) * 1e3,
            lambda detuning_mhz: current_frequency_ghz + detuning_mhz / 1e3,
        ),
    )
    detuning_axis.set_xlabel("Detuning from current resonance [MHz]")
    return detuning_axis


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
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax1, qubit in grid_iter(grid):
        selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit]
        selected.ground_phase.plot(ax=ax1, x="full_freq_GHz", label="Ground")
        selected.mixed_phase.plot(ax=ax1, x="full_freq_GHz", label="Driven")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        ax1.legend()
    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state phase")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude(ds: xr.Dataset, qubits: List[AnyTransmon]):
    """
    Plot mean resonator responses and normalized shot-cloud separation.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    locations = [
        tuple(int(value) for value in location.split(","))
        for location in qubit_grid_locations(qubits)
    ]
    rows = max(row for row, _ in locations) + 1
    columns = max(column for _, column in locations) + 1
    height_ratios = [3, 1] * rows
    fig, axes = plt.subplots(
        2 * rows,
        columns,
        figsize=FIGURE_SIZE,
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios},
    )

    used_axes = set()
    for qubit, (row, column) in zip(qubits, locations):
        spectrum_ax = axes[2 * row, column]
        difference_ax = axes[2 * row + 1, column]
        used_axes.update({(2 * row, column), (2 * row + 1, column)})

        selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).sel(qubit=qubit.name)
        ground = selected.ground_IQ_abs / u.mV
        mixed = selected.mixed_IQ_abs / u.mV
        separation = selected.IQ_separation
        max_separation_index = int(separation.argmax(dim="detuning").values)
        max_separation_frequency_ghz = float(
            selected.full_freq_GHz.isel(detuning=max_separation_index).values
        )
        current_frequency_ghz = float(qubit.resonator.RF_frequency) / u.GHz
        max_separation_label = (
            f"New resonance (maximum normalized IQ separation): {max_separation_frequency_ghz:.6f} GHz"
        )
        current_frequency_label = f"Current resonance: {current_frequency_ghz:.6f} GHz"

        ground.plot(ax=spectrum_ax, x="full_freq_GHz", label="Ground")
        mixed.plot(ax=spectrum_ax, x="full_freq_GHz", label="Driven")
        spectrum_ax.axvline(
            current_frequency_ghz,
            color="black",
            linestyle=":",
            label=current_frequency_label,
        )
        spectrum_ax.axvline(
            max_separation_frequency_ghz,
            color="tab:red",
            linestyle="--",
            label=max_separation_label,
        )
        spectrum_ax.set_title(qubit.name)
        spectrum_ax.set_xlabel("")
        spectrum_ax.set_ylabel(r"$|IQ|$ [mV]")
        spectrum_ax.legend()

        separation.plot(ax=difference_ax, x="full_freq_GHz", color="tab:blue")
        difference_ax.axvline(
            current_frequency_ghz,
            color="black",
            linestyle=":",
            label=current_frequency_label,
        )
        difference_ax.axvline(
            max_separation_frequency_ghz,
            color="tab:red",
            linestyle="--",
            label=max_separation_label,
        )
        difference_ax.set_xlabel("RF frequency [GHz]")
        difference_ax.set_ylabel("IQ separation / pooled std")
        difference_ax.legend()
        _add_detuning_axis(spectrum_ax, current_frequency_ghz)
        _add_detuning_axis(difference_ax, current_frequency_ghz)

    for row in range(2 * rows):
        for column in range(columns):
            if (row, column) not in used_axes:
                axes[row, column].set_visible(False)

    fig.suptitle("Resonator spectroscopy")
    fig.tight_layout()
    return fig


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
    (selected.mixed_IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz", label="Driven")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    if fitted_data is not None:
        ax.plot(ds.full_freq.loc[qubit] / u.GHz, fitted_data / u.mV, "k--", label="Ground fit")
    ax.legend()


def plot_iq_response(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot ground and mixed-state resonator trajectories in the IQ plane."""
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        selected = ds.loc[qubit]
        ax.plot(selected.Ig / u.mV, selected.Qg / u.mV, ".-", label="Ground", markersize=3)
        ax.plot(selected.Im / u.mV, selected.Qm / u.mV, ".-", label="Driven", markersize=3)
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.axis("equal")
        ax.legend()

    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state IQ response")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig
