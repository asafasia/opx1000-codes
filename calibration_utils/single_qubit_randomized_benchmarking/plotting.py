from utils.experiment_readout import ground_population
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


def _single_gate_error_label(fit: xr.Dataset) -> str:
    """Return a compact legend label for the fitted RB decay."""
    fidelity = float(fit.fidelity.values)
    if fidelity > 1:
        fidelity /= 100
    single_gate_error = 1 - fidelity
    return f"Fit, single gate error = {single_gate_error:.3e}"


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
    if fit is not None and fit.attrs.get("rb_mode") in {"interleaved", "leakage"}:
        return plot_mode_data(ax, fit, qubit)
    fit_succeeded = "success" in fit and bool(fit.success.values)
    if hasattr(fit, "population"):
        data = fit.population
        label = "Ground-state population"
    elif hasattr(fit, "state"):
        data = ground_population(fit)
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
        smooth_depths = xr.DataArray(
            np.linspace(float(fit.depths.min()), float(fit.depths.max()), 300),
            dims="depths",
        )
        fitted = decay_exp(
            smooth_depths,
            fit.fit_data.sel(fit_vals="a"),
            fit.fit_data.sel(fit_vals="offset"),
            fit.fit_data.sel(fit_vals="decay"),
        )
        ax.plot(smooth_depths, fitted, "r--", label=_single_gate_error_label(fit))
    else:
        ax.plot([], [], "r--", label="RB decay fit failed")
    ax.legend()


def plot_mode_data(ax, fit, qubit):
    mode = fit.attrs["rb_mode"]
    if mode == "interleaved":
        for variant in ["reference", "interleaved"]:
            data = fit.population.sel(rb_variant=variant)
            ax.errorbar(fit.depths, data.mean("nb_of_sequences"),
                        yerr=data.std("nb_of_sequences") / np.sqrt(data.sizes["nb_of_sequences"]),
                        fmt=".", label=variant.title())
            ax.plot(fit.depths, fit.predicted_population.sel(rb_variant=variant), "--")
        ax.set_ylabel("Ground-state population" if "state" in fit else "Readout signal")
        ax.set_title(f"{qubit['qubit']}: target error={float(fit.error_per_gate):.3g}")
    else:
        for state in "gef":
            data = fit[f"population_{state}"]
            ax.errorbar(fit.depths, data.mean("nb_of_sequences"),
                        yerr=data.std("nb_of_sequences") / np.sqrt(data.sizes["nb_of_sequences"]),
                        fmt=".", label=f"P({state})")
        ax.plot(fit.depths, 1 - fit.predicted_population, "--", label="Leakage fit")
        ax.set_ylabel("Population")
        ax.set_title(f"{qubit['qubit']}: L1={float(fit.leakage_per_clifford):.3g}, L2={float(fit.seepage_per_clifford):.3g} / Clifford")
    if not bool(fit.success):
        ax.text(0.02, 0.05, "Fit failed: " + str(fit.message.item()), transform=ax.transAxes, wrap=True)
    ax.set_xlabel("Random Clifford count (excluding recovery)")
    ax.grid(True)
    ax.legend()
