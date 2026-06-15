from typing import List

import matplotlib.pyplot as plt
import xarray as xr
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations


u = unit(coerce_to_integer=True)


def plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
):
    """Plot the frequency-versus-amplitude Rabi chevron."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {missing!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    return (
        _plot_state(ds, qubits)
        if use_state_discrimination
        else _plot_iq(ds, qubits)
    )


def _plot_state(ds: xr.Dataset, qubits: List[AnyTransmon]):
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit_ref in grid_iter(grid):
        selected = ds.sel(qubit=qubit_ref["qubit"]).assign_coords(
            full_freq_GHz=ds.sel(qubit=qubit_ref["qubit"]).full_freq / u.GHz,
            full_amp_mV=ds.sel(qubit=qubit_ref["qubit"]).full_amp / u.mV,
        )
        plotted = selected["state"].plot(
            ax=ax,
            x="full_freq_GHz",
            y="full_amp_mV",
            add_colorbar=True,
            robust=True,
        )
        label = "Measured state"
        plotted.colorbar.set_label(label)
        ax.set_title(f"{qubit_ref['qubit']}: {label}")
        ax.set_xlabel("RF frequency [GHz]")
        ax.set_ylabel("Pulse amplitude [mV]")

    grid.fig.suptitle("Power Rabi chevron: state")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def _plot_iq(ds: xr.Dataset, qubits: List[AnyTransmon]):
    num_qubits = len(qubits)
    figure, axes = plt.subplots(
        num_qubits,
        2,
        squeeze=False,
        figsize=(FIGURE_SIZE[0] * 2, FIGURE_SIZE[1] * max(1, num_qubits)),
    )

    for row, qubit in enumerate(qubits):
        qubit_name = qubit.name
        selected = ds.sel(qubit=qubit_name).assign_coords(
            full_freq_GHz=ds.sel(qubit=qubit_name).full_freq / u.GHz,
            full_amp_mV=ds.sel(qubit=qubit_name).full_amp / u.mV,
        )
        for column, variable in enumerate(("I", "Q")):
            ax = axes[row][column]
            plotted = (selected[variable] / u.mV).plot(
                ax=ax,
                x="full_freq_GHz",
                y="full_amp_mV",
                add_colorbar=True,
                robust=True,
            )
            label = f"{variable} [mV]"
            plotted.colorbar.set_label(label)
            ax.set_title(f"{qubit_name}: {label}")
            ax.set_xlabel("RF frequency [GHz]")
            ax.set_ylabel("Pulse amplitude [mV]")

    figure.suptitle("Power Rabi chevron: I and Q quadratures")
    figure.tight_layout()
    return figure
