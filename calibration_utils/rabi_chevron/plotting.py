from typing import List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE

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
    - State-discrimination mode creates one subplot per qubit.
    - Analog-readout mode creates separate I and Q subplots per qubit.
    """
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Rabi-chevron plot expected {variables} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )

    fig, axes = plt.subplots(
        len(variables) * len(qubits),
        1,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )
    for qubit_index, qubit in enumerate(qubits):
        for variable_index, variable in enumerate(variables):
            plot_individual_data_with(
                axes[len(variables) * qubit_index + variable_index, 0],
                ds,
                {"qubit": qubit.name},
                fits.sel(qubit=qubit.name),
                use_state_discrimination=use_state_discrimination,
                variable=variable,
            )

    fig.suptitle(
        "Rabi chevron: measured state"
        if use_state_discrimination
        else "Rabi chevron: I and Q quadratures"
    )
    fig.tight_layout()
    return fig


def plot_individual_data_with(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    use_state_discrimination: bool = False,
    variable: str | None = None,
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

    data = variable or ("state" if use_state_discrimination else "I")
    expected_variables = ("state",) if use_state_discrimination else ("I", "Q")
    if data not in expected_variables:
        raise ValueError(
            f"Rabi-chevron variable {data!r} is incompatible with "
            f"use_state_discrimination={use_state_discrimination}"
        )
    if data not in ds:
        raise RuntimeError(
            f"Rabi-chevron plot expected {data!r} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )
    scale = 1 if data == "state" else 1 / u.mV
    data_label = "Measured state" if data == "state" else f"{data} [mV]"

    # Create a first x-axis for full_freq_GHz
    plotted = (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz)[data] * scale).plot(
        ax=ax, y="pulse_duration", x="full_freq_GHz", add_colorbar=True
    )
    plotted.colorbar.set_label(data_label)
    ax.set_title(f"{qubit['qubit']}: {data_label}")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Pulse duration [ns]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz)[data] * scale).plot(
        ax=ax2, y="pulse_duration", x="detuning_MHz", add_colorbar=False
    )
    ax2.set_title("")
    ax2.set_xlabel("Detuning [MHz]")
