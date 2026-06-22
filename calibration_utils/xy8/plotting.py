from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.analysis import decay_exp
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("XY8 decay vs. total evolution time")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    selected = ds.sel(qubit=qubit["qubit"])
    for n_xy8 in selected.n_xy8.values:
        evolution_time = selected.evolution_time
        if "state" in selected:
            y = selected.state.sel(n_xy8=n_xy8)
            scale = 1.0
        else:
            quadrature = str(fit.selected_quadrature.sel(n_xy8=n_xy8).values)
            y = selected[quadrature].sel(n_xy8=n_xy8)
            scale = 1e3
        ax.plot(
            1e-3 * evolution_time,
            scale * y,
            ".",
            label=f"N={int(n_xy8)}",
        )
        if fit is not None and np.isfinite(fit.fit_data.sel(n_xy8=n_xy8)).all():
            fitted = decay_exp(
                evolution_time,
                fit.fit_data.sel(n_xy8=n_xy8, fit_vals="a"),
                fit.fit_data.sel(n_xy8=n_xy8, fit_vals="offset"),
                fit.fit_data.sel(n_xy8=n_xy8, fit_vals="decay"),
            )
            ax.plot(1e-3 * evolution_time, scale * fitted, "-")

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Total evolution time [us]")
    ax.set_ylabel("State" if "state" in selected else "Selected quadrature [mV]")
    ax.legend(loc="best", fontsize="small")


def plot_t2_vs_order(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    grid = QubitGrid(fits, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        selected = fits.sel(qubit=qubit["qubit"])
        ax.errorbar(
            selected.n_xy8,
            1e-3 * selected.T2_xy8,
            yerr=1e-3 * selected.T2_xy8_error,
            marker="o",
            linestyle="-",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("XY8 cycles N")
        ax.set_ylabel("T2 XY8 [us]")
        ax.set_title(qubit["qubit"])
        ax.grid(True, alpha=0.3)

    grid.fig.suptitle("XY8 coherence time vs. decoupling order")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig
