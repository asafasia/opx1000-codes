from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.fine_rabi.parameters import operation_for_rotation
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
    operation = operation_for_rotation(rotation_type)

    figure, axes = plt.subplots(
        len(qubits) * 2,
        1,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * max(1, len(qubits))),
        sharex=True,
        squeeze=False,
    )
    for row, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name)
        selected_fit = fits.sel(qubit=qubit.name) if fits is not None else None
        data = selected[data_name]
        operation_amplitude = _operation_amplitude(qubit, operation)
        scan_ax = axes[2 * row, 0]
        fourier_ax = axes[2 * row + 1, 0]
        _plot_scan(scan_ax, data, data_label, qubit.name, selected_fit, operation_amplitude)
        _plot_fourier(fourier_ax, data, qubit.name, selected_fit, operation_amplitude)

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


def _operation_amplitude(qubit: AnyTransmon, operation: str) -> float | None:
    try:
        return float(qubit.xy.operations[operation].amplitude)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _plot_scan(
    ax: Axes,
    data: xr.DataArray,
    data_label: str,
    qubit_name: str,
    fit: xr.Dataset | None,
    operation_amplitude: float | None,
) -> None:
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
    _plot_amplitude_markers(ax, fit)
    _add_real_amplitude_axis(ax, operation_amplitude)
    ax.legend(loc="upper right", fontsize=8)


def _plot_fourier(
    ax: Axes,
    data: xr.DataArray,
    qubit_name: str,
    fit: xr.Dataset | None,
    operation_amplitude: float | None,
) -> None:
    fft_data = _fourier_data_from_fit_or_raw(data, fit)
    plotted = fft_data.plot(
        ax=ax,
        x="amp_prefactor",
        y="fourier_frequency",
        add_colorbar=True,
        robust=True,
        cmap="magma",
    )
    plotted.colorbar.set_label("Fourier amplitude")
    ax.set_title(f"{qubit_name}: Fourier analysis")
    ax.set_xlabel("Pulse amplitude factor")
    ax.set_ylabel("Frequency [cycles/group]")
    _plot_amplitude_markers(ax, fit)
    _add_real_amplitude_axis(ax, operation_amplitude)
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


def _plot_amplitude_markers(ax: Axes, fit: xr.Dataset | None) -> None:
    ax.axvline(
        1.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        label="Current amplitude",
    )
    if fit is not None and "optimal_amp_prefactor" in fit:
        optimum = float(fit.optimal_amp_prefactor)
        if np.isfinite(optimum):
            ax.axvline(
                optimum,
                color="tab:green",
                linestyle="-",
                linewidth=1.6,
                label="Optimized amplitude",
            )


def _add_real_amplitude_axis(ax: Axes, operation_amplitude: float | None) -> None:
    if operation_amplitude is None or not np.isfinite(operation_amplitude) or operation_amplitude == 0:
        return
    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda prefactor: prefactor * operation_amplitude * 1e3,
            lambda amplitude_mv: amplitude_mv / (operation_amplitude * 1e3),
        ),
    )
    secondary.set_xlabel("Pulse amplitude [mV]")


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
    for branch, linestyle in (("left", "-"), ("right", "--")):
        slope = float(fit.branch_line_coefficients.sel(branch=branch, line_parameter="slope"))
        intercept = float(fit.branch_line_coefficients.sel(branch=branch, line_parameter="intercept"))
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color="tab:blue",
            linestyle=linestyle,
            linewidth=2,
            label=f"{branch.capitalize()} linear fit",
            zorder=4,
        )
    optimum = float(fit.optimal_amp_prefactor)
    optimum_frequency = float(fit.optimal_frequency)
    ax.scatter(
        [optimum],
        [optimum_frequency],
        marker="x",
        s=70,
        color="tab:green",
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
