from typing import List
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    variables = ("I", "Q") if "I" in ds and "Q" in ds else ("state",)
    fig, axes = plt.subplots(
        len(qubits) * len(variables),
        1,
        figsize=(10, max(4, 4 * len(qubits) * len(variables))),
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

    fig.suptitle("Power Rabi: I and Q quadratures")
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
            fitted_data = oscillation(
                fit.amp_prefactor.data,
                fit.fit.sel(fit_vals="a").data,
                fit.fit.sel(fit_vals="f").data,
                fit.fit.sel(fit_vals="phi").data,
                fit.fit.sel(fit_vals="offset").data,
            )
        else:
            fitted_data = None

        selected = ds
        if "nb_of_pulses" in selected:
            selected = selected.isel(nb_of_pulses=0, drop=True)
        selected = selected.assign_coords(amp_mV=selected.full_amp * 1e3)
        scale = 1 if variable == "state" else 1e3
        (selected[variable] * scale).plot(ax=ax, x="amp_mV")
        if fitted_data is not None and variable in ("I", "state"):
            ax.plot(fit.full_amp * 1e3, scale * fitted_data, "r--", label="Fit")
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
