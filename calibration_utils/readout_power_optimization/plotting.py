from typing import List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

u = unit(coerce_to_integer=True)

TRACE_SPECS = {
    "meas_fidelity": {
        "label": "Assignment fidelity",
        "color": "tab:blue",
        "marker": "o",
    },
    "outliers": {
        "label": "Non-outlier probability",
        "color": "tab:orange",
        "marker": "s",
    },
}


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
    locations = [
        tuple(int(value) for value in location.split(","))
        for location in qubit_grid_locations(qubits)
    ]
    rows = max(row for row, _ in locations) + 1
    columns = max(column for _, column in locations) + 1
    fig, axes = _make_probability_and_separation_axes(rows, columns)

    used_axes = set()
    last_probability_ax = None
    last_separation_ax = None
    for qubit, (row, column) in zip(qubits, locations):
        probability_ax = axes[2 * row, column]
        separation_ax = axes[2 * row + 1, column]
        used_axes.update({(2 * row, column), (2 * row + 1, column)})
        qubit_ref = {"qubit": qubit.name}
        selected_ds = ds.sel(qubit=qubit.name)
        selected_fit = fits.sel(qubit=qubit.name)
        plot_individual_data_with_fit(probability_ax, selected_ds, qubit_ref, selected_fit)
        plot_individual_separation(separation_ax, selected_ds, qubit_ref, selected_fit)
        probability_ax.set_xlabel("")
        probability_ax.tick_params(labelbottom=False)
        last_probability_ax = probability_ax
        last_separation_ax = separation_ax

    for row in range(2 * rows):
        for column in range(columns):
            if (row, column) not in used_axes:
                axes[row, column].set_visible(False)

    handles, labels = [], []
    for source_ax in (last_probability_ax, last_separation_ax):
        if source_ax is None:
            continue
        source_handles, source_labels = source_ax.get_legend_handles_labels()
        handles.extend(source_handles)
        labels.extend(source_labels)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4)
    fig.suptitle("Readout power optimization")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    return fig


def _make_probability_and_separation_axes(rows: int, columns: int):
    return plt.subplots(
        2 * rows,
        columns,
        figsize=FIGURE_SIZE,
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1.25] * rows},
    )


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
    x = _readout_amplitude_mv(fit)
    shot_count = _shot_count(ds)
    for fit_value in fit.fit_vals.values:
        name = str(fit_value)
        spec = TRACE_SPECS.get(name, {"label": name, "color": None, "marker": "o"})
        y = np.asarray(fit.fit_data.sel(fit_vals=fit_value).values, dtype=float)
        yerr = _binomial_error(y, shot_count)
        color = spec["color"]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=spec["marker"],
            markersize=5,
            linewidth=1.8,
            capsize=3,
            color=color,
            label=spec["label"],
        )
        ax.fill_between(
            x,
            np.clip(y - yerr, 0, 1),
            np.clip(y + yerr, 0, 1),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    optimal_amp_mv = 1e3 * float(fit.optimal_amp)
    ax.axvline(
        optimal_amp_mv,
        color="0.15",
        linestyle="--",
        linewidth=1.2,
        label=f"Optimal readout amplitude ({optimal_amp_mv:.3g} mV)",
    )
    if "best_fidelity" in fit:
        best_fidelity = float(fit.best_fidelity)
        ax.plot(
            optimal_amp_mv,
            best_fidelity,
            marker="*",
            markersize=13,
            color="tab:green",
            markeredgecolor="0.15",
            label=f"Best fidelity {100 * best_fidelity:.1f}%",
        )

    ax.set_xlabel("Readout amplitude [mV]")
    ax.set_ylabel("Probability")
    ax.set_title(qubit["qubit"])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize="small")


def plot_individual_separation(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot IQ-center separation below the readout-power fidelity panel."""
    if "I" not in ds or "Q" not in ds:
        ax.text(0.5, 0.5, "No IQ data available", ha="center", va="center", color="0.45")
        ax.set_axis_off()
        return

    x = _readout_amplitude_mv(fit)
    separation, separation_error = _iq_center_separation_mv(ds)
    ax.errorbar(
        x,
        separation,
        yerr=separation_error,
        marker="D",
        markersize=4,
        linewidth=1.5,
        capsize=3,
        color="tab:purple",
        label="IQ center separation",
    )
    optimal_amp_mv = 1e3 * float(fit.optimal_amp)
    ax.axvline(optimal_amp_mv, color="0.15", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Readout amplitude [mV]")
    ax.set_ylabel("Separation [mV]")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize="small")


def _readout_amplitude_mv(fit: xr.Dataset) -> np.ndarray:
    if "readout_amplitude" in fit.coords:
        return 1e3 * np.asarray(fit.readout_amplitude.values, dtype=float)
    return np.asarray(fit.amp_prefactor.values, dtype=float)


def _shot_count(ds: xr.Dataset) -> int:
    state_count = int(ds.sizes.get("state", 2))
    run_count = int(ds.sizes.get("n_runs", 1))
    return max(state_count * run_count, 1)


def _binomial_error(probability: np.ndarray, shot_count: int) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    return np.sqrt(probability * (1 - probability) / shot_count)


def _iq_center_separation_mv(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    ground_i = ds.I.sel(state=0)
    excited_i = ds.I.sel(state=1)
    ground_q = ds.Q.sel(state=0)
    excited_q = ds.Q.sel(state=1)
    delta_i = excited_i.mean(dim="n_runs") - ground_i.mean(dim="n_runs")
    delta_q = excited_q.mean(dim="n_runs") - ground_q.mean(dim="n_runs")
    separation = np.hypot(delta_i, delta_q)

    run_count = max(int(ds.sizes.get("n_runs", 1)), 1)
    var_delta_i = (ground_i.var(dim="n_runs") + excited_i.var(dim="n_runs")) / run_count
    var_delta_q = (ground_q.var(dim="n_runs") + excited_q.var(dim="n_runs")) / run_count
    safe_separation = separation.where(separation > 0, other=np.nan)
    unit_i = delta_i / safe_separation
    unit_q = delta_q / safe_separation
    separation_var = (unit_i**2 * var_delta_i + unit_q**2 * var_delta_q).fillna(0)
    return (
        1e3 * np.asarray(separation.values, dtype=float),
        1e3 * np.sqrt(np.asarray(separation_var.values, dtype=float)),
    )
