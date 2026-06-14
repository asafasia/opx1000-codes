from typing import List

import xarray as xr
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE


u = unit(coerce_to_integer=True)


def plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
):
    """Plot the frequency-versus-amplitude Rabi chevron."""
    variable = "state" if use_state_discrimination else "I"
    if variable not in ds:
        raise RuntimeError(
            f"Power-Rabi-chevron plot expected {variable!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    grid = QubitGrid(ds, [qubit.grid_location for qubit in qubits])
    for ax, qubit_ref in grid_iter(grid):
        selected = ds.sel(qubit=qubit_ref["qubit"]).assign_coords(
            full_freq_GHz=ds.sel(qubit=qubit_ref["qubit"]).full_freq / u.GHz,
            full_amp_mV=ds.sel(qubit=qubit_ref["qubit"]).full_amp / u.mV,
        )
        scale = 1 if variable == "state" else 1 / u.mV
        plotted = (selected[variable] * scale).plot(
            ax=ax,
            x="full_freq_GHz",
            y="full_amp_mV",
            add_colorbar=True,
            robust=True,
        )
        label = "Measured state" if variable == "state" else "I [mV]"
        plotted.colorbar.set_label(label)
        ax.set_title(f"{qubit_ref['qubit']}: {label}")
        ax.set_xlabel("RF frequency [GHz]")
        ax.set_ylabel("Pulse amplitude [mV]")

    grid.fig.suptitle(
        "Power Rabi chevron: state"
        if use_state_discrimination
        else "Power Rabi chevron: I quadrature"
    )
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig
