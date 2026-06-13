from typing import List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE

u = unit(coerce_to_integer=True)


def ideal_state_response(amp_prefactor, number_of_pulses: int, operation: str):
    """Return the ideal excited-state population for repeated rotations."""
    rotation = np.pi if operation.endswith("x180") or operation.startswith("x180_") else np.pi / 2
    return np.sin(number_of_pulses * rotation * amp_prefactor / 2) ** 2


def expected_cycles_to_unit_prefactor(number_of_pulses: int, operation: str) -> float:
    """Return ideal full population cycles between amplitude prefactors 0 and 1."""
    return number_of_pulses / (
        2 if operation.endswith("x180") or operation.startswith("x180_") else 4
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    use_state_discrimination: bool = False,
):
    """
    Plot power-Rabi I and Q responses in vertically stacked subplots.

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
    - Each qubit occupies two rows containing separate I and Q subplots.
    - State-discriminated datasets use one subplot per qubit.
    """
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Power-Rabi plot expected {variables} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )
    fig, axes = plt.subplots(
        len(qubits) * len(variables),
        1,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )

    axis_index = 0
    for qubit in qubits:
        selected = ds.sel(qubit=qubit.name)
        fit = fits.sel(qubit=qubit.name)
        for variable in variables:
            ax = axes[axis_index, 0]
            if "nb_of_pulses" not in ds or len(ds.nb_of_pulses) == 1:
                plot_individual_data_with_fit_1D(ax, selected, variable, fit)
            else:
                plot_individual_data_with_fit_2D(ax, selected, variable, fit)
            ax.set_title(f"{qubit.name}: {variable}")
            axis_index += 1

    fig.suptitle("Power Rabi: state" if use_state_discrimination else "Power Rabi: I and Q quadratures")
    fig.tight_layout()
    return fig


def plot_individual_data_with_fit_1D(ax: Axes, ds: xr.Dataset, variable: str, fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    variable : str
        Dataset variable to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    if "nb_of_pulses" not in ds or len(ds.nb_of_pulses.data) == 1:
        if fit is not None:
            fit_variable = f"fit_{variable}" if f"fit_{variable}" in fit else "fit"
            channel_fit = fit[fit_variable]
            fitted_data = oscillation(
                fit.amp_prefactor.data,
                channel_fit.sel(fit_vals="a").data,
                channel_fit.sel(fit_vals="f").data,
                channel_fit.sel(fit_vals="phi").data,
                channel_fit.sel(fit_vals="offset").data,
            )
        else:
            fitted_data = None

        selected = ds
        if "nb_of_pulses" in selected:
            selected = selected.isel(nb_of_pulses=0, drop=True)
        selected = selected.assign_coords(amp_mV=selected.full_amp * 1e3)
        scale = 1 if variable == "state" else 1e3
        (selected[variable] * scale).plot(ax=ax, x="amp_mV")
        if variable == "state":
            number_of_pulses = (
                int(np.asarray(ds.nb_of_pulses.values).flat[0])
                if "nb_of_pulses" in ds.coords
                else 1
            )
            operation = str(fit.operation.values) if "operation" in fit else "x180"
            ideal_state = ideal_state_response(
                selected.amp_prefactor,
                number_of_pulses,
                operation,
            )
            expected_cycles = expected_cycles_to_unit_prefactor(number_of_pulses, operation)
            ax.plot(
                selected.amp_mV,
                ideal_state,
                "k--",
                alpha=0.6,
                label=(
                    f"Ideal: {number_of_pulses} {operation} gates "
                    f"({expected_cycles:g} cycles from prefactor 0 to 1)"
                ),
            )
        if fitted_data is not None:
            selected = "selected_quadrature" in fit and str(fit.selected_quadrature.values) == variable
            score_name = f"r_squared_{variable}"
            score = float(fit[score_name].values) if score_name in fit else None
            label = "Fit"
            if score is not None:
                label += f" ($R^2$={score:.3f})"
            if selected:
                label += " - selected"
            ax.plot(fit.full_amp * 1e3, scale * fitted_data, "r--", label=label)
            ax.legend()
        ax.set_ylabel("Qubit state" if variable == "state" else f"{variable} [mV]")
        ax.set_xlabel("Pulse amplitude [mV]")
        ax.grid(alpha=0.25)


def plot_individual_data_with_fit_2D(ax: Axes, ds: xr.Dataset, variable: str, fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    variable : str
        Dataset variable to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ds.assign_coords(amp_mV=ds.full_amp * 1e3)[variable].plot(
        ax=ax, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax.set_ylabel("Number of pulses")
    ax.set_xlabel("Pulse amplitude [mV]")
    if bool(fit.success.values):
        ax.axvline(
            x=float(fit.opt_amp.values) * 1e3,
            color="g",
            linestyle="-",
            label="Optimal amplitude",
        )
        ax.legend()
