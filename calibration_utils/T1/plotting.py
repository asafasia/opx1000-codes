from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.plotting import QubitGrid, grid_iter
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots T1 data with fit results for multiple qubits.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    node_parameters : Parameters
        Parameters related to the node.
    fits : xr.Dataset
        Dataset containing the fit results for the T1 data.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    grid = QubitGrid(ds, qubit_grid_locations(qubits))

    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle(f"{fits.attrs.get('t1_label', 'T1')} vs. idle time")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit data on a given axis."""
    if _fit_is_plottable(fit):
        fitted = decay_exp(
            ds.idle_time,
            fit.fit_data.sel(fit_vals="a"),
            fit.fit_data.sel(fit_vals="offset"),
            fit.fit_data.sel(fit_vals="decay"),
        )
    else:
        fitted = None

    if hasattr(fit, "state"):
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax, marker=".", linestyle="-", markersize=5)
        if fitted is not None:
            ax.plot(ds.idle_time, fitted, "r--")
            ax.axhline(
                float(fit.fit_data.sel(fit_vals="offset").values),
                color="black",
                linestyle="--",
                label="B",
            )
        ax.set_ylabel("State")
        ax.set_ylim(0, 1)
    elif hasattr(fit, "I"):
        quadrature = str(fit.selected_quadrature.values) if "selected_quadrature" in fit else "I"
        (ds.sel(qubit=qubit["qubit"])[quadrature] * 1e3).plot(
            ax=ax,
            marker=".",
            linestyle="-",
            markersize=5,
        )
        if fitted is not None:
            ax.plot(ds.idle_time, fitted * 1e3, "r--")
            ax.axhline(
                1e3 * float(fit.fit_data.sel(fit_vals="offset").values),
                color="black",
                linestyle="--",
                label="B",
            )
        ax.set_ylabel(f"Trans. amp. {quadrature} [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        _add_fit_text(ax, fit)


def _fit_is_plottable(fit: xr.Dataset | None) -> bool:
    """Return True only when a T1 fit should be drawn as a fitted curve."""
    if fit is None or "fit_data" not in fit:
        return False
    if "success" in fit and not bool(fit.success.values):
        return False
    required_values = fit.fit_data.sel(fit_vals=["a", "offset", "decay"]).values
    return bool(np.isfinite(required_values).all())


def _add_fit_text(ax, fit):
    """Add fit results text to the axis."""
    ax.text(
        0.1,
        0.9,
        f"{fit.attrs.get('t1_label', 'T1')} = {1e-3 * fit.tau.values:.1f} ± {1e-3 * fit.tau_error.values:.1f} µs\nSuccess: {fit.success.values}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )


def plot_population(ds: xr.Dataset, qubits: List[AnyTransmon], *, population: str):
    """Plot an individual state population versus idle time without a decay fit."""
    if population not in ("g", "e", "f"):
        raise ValueError("Population must be g, e, or f.")
    variable = f"population_{population}"
    if variable not in ds:
        raise RuntimeError(f"T1 population plot requires {variable!r}.")
    index = "gef".index(population)
    label = f"P{index} ({population}-state population)"
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        selected = ds.sel(qubit=qubit["qubit"])
        ax.plot(
            selected.idle_time, selected[variable],
            marker=".", linestyle="-", markersize=5, color=f"C{index}",
        )
        ax.set_title(f"{qubit['qubit']}: {label}")
        ax.set_xlabel("Idle time [ns]")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    grid.fig.suptitle(f"T1: {label} vs. idle time")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig
