from __future__ import annotations

import numpy as np
import xarray as xr

from qualibration_libs.plotting import QubitGrid, grid_iter
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations


TRACE_SCALE = 1e6
TRACE_UNIT = "uV"


def plot_readout_weight_traces(ds: xr.Dataset, qubits):
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        slot = ax.get_subplotspec()
        ax.remove()
        subgrid = slot.subgridspec(
            2,
            2,
            height_ratios=[2, 1],
            hspace=0.22,
            wspace=0.25,
        )
        iq_ax = grid.fig.add_subplot(subgrid[0, 0])
        magnitude_ax = grid.fig.add_subplot(subgrid[0, 1])
        kernel_ax = grid.fig.add_subplot(subgrid[1, :], sharex=magnitude_ax)

        selected = ds.sel(qubit=qubit["qubit"])
        iq_ax.plot(
            TRACE_SCALE * selected.ground_trace.real,
            TRACE_SCALE * selected.ground_trace.imag,
            ".-",
            label="g",
        )
        iq_ax.plot(
            TRACE_SCALE * selected.excited_trace.real,
            TRACE_SCALE * selected.excited_trace.imag,
            ".-",
            label="e",
        )
        iq_ax.set_title(f"{qubit['qubit']} IQ trace")
        iq_ax.set_xlabel(f"I [{TRACE_UNIT}]")
        iq_ax.set_ylabel(f"Q [{TRACE_UNIT}]")
        iq_ax.legend(loc="best")

        magnitude_ax.plot(
            selected.time_ns,
            TRACE_SCALE * np.abs(selected.ground_trace),
            label="|g|",
        )
        magnitude_ax.plot(
            selected.time_ns,
            TRACE_SCALE * np.abs(selected.excited_trace),
            label="|e|",
        )
        magnitude_ax.set_title(f"{qubit['qubit']} magnitude")
        magnitude_ax.set_ylabel(f"|I + iQ| [{TRACE_UNIT}]")
        magnitude_ax.legend(loc="best")
        magnitude_ax.tick_params(labelbottom=False)

        kernel_ax.plot(selected.time_ns, selected.profile_kernel, color="black", label="profile kernel")
        kernel_ax.axhline(0, color="0.6", linewidth=0.8)
        kernel_ax.set_xlabel("Readout time [ns]")
        kernel_ax.set_ylabel("Kernel")
        kernel_ax.legend(loc="best")

    grid.fig.suptitle("Readout weights optimization")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig
