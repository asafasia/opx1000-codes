from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE


def plot_fine_rabi(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool,
    rotation_type: str,
    fits: xr.Dataset | None = None,
) -> Figure:
    """Plot fine Rabi amplitude scan and repetition-axis Fourier map."""
    data_name = _select_data_name(ds, use_state_discrimination)
    data_label = "Measured state" if data_name == "state" else "I [V]"

    figure, axes = plt.subplots(
        len(qubits),
        2,
        figsize=(FIGURE_SIZE[0] * 2, FIGURE_SIZE[1] * max(1, len(qubits))),
        squeeze=False,
    )
    for row, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name)
        selected_fit = fits.sel(qubit=qubit.name) if fits is not None else None
        data = selected[data_name]
        _plot_scan(axes[row, 0], data, data_label, qubit.name)
        _plot_fourier(axes[row, 1], data, qubit.name, selected_fit)

    figure.suptitle(f"Fine Rabi calibration: {rotation_type}")
    figure.tight_layout()
    return figure


def _select_data_name(ds: xr.Dataset, use_state_discrimination: bool) -> str:
    if use_state_discrimination:
        if "state" in ds:
            return "state"
        raise RuntimeError(f"Fine-Rabi plot expected state data, but dataset contains {list(ds.data_vars)}")
    if "I" not in ds:
        raise RuntimeError(f"Fine-Rabi plot expected I/Q data, but dataset contains {list(ds.data_vars)}")
    return "I"


def _plot_scan(ax, data: xr.DataArray, data_label: str, qubit_name: str) -> None:
    plotted = data.plot(
        ax=ax,
        x="amp_prefactor",
        y="repetition_group_count",
        add_colorbar=True,
        robust=True,
    )
    plotted.colorbar.set_label(data_label)
    ax.set_title(f"{qubit_name}: Fine Rabi scan")
    ax.set_xlabel("Pulse amplitude factor")
    ax.set_ylabel("Repetition group count")


def _plot_fourier(ax, data: xr.DataArray, qubit_name: str, fit: xr.Dataset | None) -> None:
    fft_data = _fourier_data_from_fit_or_raw(data, fit)
    plotted = fft_data.plot(
        ax=ax,
        x="amp_prefactor",
        y="fourier_frequency",
        add_colorbar=True,
        robust=True,
    )
    plotted.colorbar.set_label("Fourier amplitude")
    ax.set_title(f"{qubit_name}: Fourier analysis")
    ax.set_xlabel("Pulse amplitude factor")
    ax.set_ylabel("Frequency [cycles/group]")
    if fit is not None and "ridge_frequency" in fit:
        _plot_fourier_fit_overlay(ax, fit)


def _fourier_data_from_fit_or_raw(data: xr.DataArray, fit: xr.Dataset | None) -> xr.DataArray:
    if fit is not None and "fourier_amplitude" in fit:
        return fit.fourier_amplitude
    repetition_counts = data.repetition_group_count.values
    if repetition_counts.size < 2:
        raise ValueError("Fourier analysis requires at least two repetition points.")
    spacing = float(np.mean(np.diff(repetition_counts)))
    centered = data - data.mean(dim="repetition_group_count")
    fft_values = np.abs(np.fft.rfft(centered.values, axis=0))
    frequencies = np.fft.rfftfreq(repetition_counts.size, d=spacing)
    return xr.DataArray(
        fft_values,
        dims=("fourier_frequency", "amp_prefactor"),
        coords={
            "fourier_frequency": frequencies,
            "amp_prefactor": data.amp_prefactor.values,
        },
    )


def _plot_fourier_fit_overlay(ax, fit: xr.Dataset) -> None:
    amp_prefactors = fit.amp_prefactor.values
    ridge_frequency = fit.ridge_frequency.values
    left_mask = fit.left_branch_mask.values.astype(bool)
    right_mask = fit.right_branch_mask.values.astype(bool)
    ax.scatter(
        amp_prefactors[left_mask],
        ridge_frequency[left_mask],
        s=24,
        color="white",
        edgecolor="black",
        linewidth=0.6,
        label="Detected left branch",
        zorder=3,
    )
    ax.scatter(
        amp_prefactors[right_mask],
        ridge_frequency[right_mask],
        s=24,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        label="Detected right branch",
        zorder=3,
    )
    line_x = np.linspace(float(amp_prefactors.min()), float(amp_prefactors.max()), 200)
    for branch, color in (("left", "white"), ("right", "tab:red")):
        slope = float(fit.branch_line_coefficients.sel(branch=branch, line_parameter="slope"))
        intercept = float(fit.branch_line_coefficients.sel(branch=branch, line_parameter="intercept"))
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color=color,
            linewidth=2,
            label=f"{branch.capitalize()} linear fit",
            zorder=4,
        )
    optimum = float(fit.optimal_amp_prefactor)
    optimum_frequency = float(fit.optimal_frequency)
    ax.axvline(optimum, color="black", linestyle="--", linewidth=1.5, label="Optimum")
    ax.scatter(
        [optimum],
        [optimum_frequency],
        marker="x",
        s=70,
        color="black",
        linewidth=2,
        zorder=5,
    )
    ax.text(
        0.02,
        0.96,
        f"opt amp = {optimum:.6g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=8)
