from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation_decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE

u = unit(coerce_to_integer=True)


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
    fig, axes = plt.subplots(
        2 * len(qubits),
        1,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * len(qubits)),
        squeeze=False,
        sharex=False,
    )
    for row, qubit in enumerate(qubits):
        qubit_name = qubit.name
        qubit_ds = _select_qubit(ds, qubit_name)
        qubit_fit = _select_qubit(fits, qubit_name)
        plot_individual_data_with_fit(
            axes[2 * row, 0],
            qubit_ds,
            {"qubit": qubit_name},
            qubit_fit,
        )
        plot_fft_peak(axes[2 * row + 1, 0], qubit_ds, qubit_fit)

    fig.suptitle("Ramsey oscillations: cosine fit and FFT peak")
    fig.tight_layout()
    return fig


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
    if fit is not None:
        fitted_ramsey_data = oscillation_decay_exp(
            ds.idle_time,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
            fit.sel(fit_vals="decay"),
        )
    else:
        fitted_ramsey_data = None

    if hasattr(ds, "state"):
        plot_state(ax, ds, qubit, fitted_ramsey_data)
        ax.set_ylabel("State Population")
    elif hasattr(ds, "I"):
        plot_transmission_amplitude(ax, ds, qubit, fitted_ramsey_data)
        ax.set_ylabel("Trans. amp. I [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(f"{qubit['qubit']} Ramsey fit")
    if fit is not None:
        add_fit_text(ax, fit)
    ax.legend()


def plot_state(ax, ds, qubit, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(detuning_signs=1).state.plot(ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label=r"$\Delta$ = +")
    ds.sel(detuning_signs=-1).state.plot(ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label=r"$\Delta$ = -")
    if fitted is not None:
        ax.plot(
            ds.idle_time,
            fitted.fit.sel(detuning_signs=1),
            c="C0",
            ls="-",
            lw=1,
        )
        ax.plot(
            ds.idle_time,
            fitted.fit.sel(detuning_signs=-1),
            c="C1",
            ls="-",
            lw=1,
        )


def plot_transmission_amplitude(ax, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(detuning_signs=1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label=r"$\Delta$ = +"
    )
    (ds.sel(detuning_signs=-1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label=r"$\Delta$ = -"
    )
    if fitted is not None:
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=1), c="C0", ls="-", lw=1)
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=-1), c="C1", ls="-", lw=1)


def plot_fft_peak(ax: Axes, ds: xr.Dataset, fit: xr.Dataset | None = None):
    """Plot the Ramsey FFT spectrum and mark the strongest oscillation frequency."""
    colors = {1: "C0", -1: "C1"}
    peak_labels = []
    for detuning_sign in (1, -1):
        frequency_mhz, amplitude = ramsey_fft_spectrum(ds, detuning_sign)
        if frequency_mhz.size == 0:
            continue

        peak_index = int(np.argmax(amplitude))
        peak_frequency_mhz = float(frequency_mhz[peak_index])
        peak_amplitude = float(amplitude[peak_index])
        label = f"$\\Delta$ = {detuning_sign:+d}, peak {peak_frequency_mhz:.3f} MHz"
        ax.plot(frequency_mhz, amplitude, color=colors[detuning_sign], lw=1.2, label=label)
        ax.axvline(peak_frequency_mhz, color=colors[detuning_sign], ls="--", lw=0.9, alpha=0.75)
        ax.plot(peak_frequency_mhz, peak_amplitude, marker="o", color=colors[detuning_sign], ms=4)
        peak_labels.append(f"{detuning_sign:+d}: {peak_frequency_mhz:.3f} MHz")

    ax.set_title("FFT oscillation spectrum")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("FFT amplitude [a.u.]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if peak_labels:
        ax.text(
            0.99,
            0.95,
            "Max resonance\n" + "\n".join(peak_labels),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"),
        )


def ramsey_fft_spectrum(ds: xr.Dataset, detuning_sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-frequency FFT bins in MHz for one detuning sign."""
    signal = _signal_for_fft(ds, detuning_sign)
    idle_time = np.asarray(ds.idle_time, dtype=float)
    if signal.size < 2 or idle_time.size < 2:
        return np.array([]), np.array([])

    order = np.argsort(idle_time)
    idle_time = idle_time[order]
    signal = signal[order]
    valid = np.isfinite(idle_time) & np.isfinite(signal)
    idle_time = idle_time[valid]
    signal = signal[valid]
    if signal.size < 2:
        return np.array([]), np.array([])

    dt_ns = float(np.median(np.diff(idle_time)))
    if not np.isfinite(dt_ns) or dt_ns <= 0:
        return np.array([]), np.array([])

    centered = signal - np.mean(signal)
    frequency_mhz = np.fft.rfftfreq(centered.size, d=dt_ns) * 1e3
    amplitude = np.abs(np.fft.rfft(centered))
    if frequency_mhz.size <= 1:
        return np.array([]), np.array([])
    return frequency_mhz[1:], amplitude[1:]


def _signal_for_fft(ds: xr.Dataset, detuning_sign: int) -> np.ndarray:
    selected = ds.sel(detuning_signs=detuning_sign)
    if hasattr(selected, "state"):
        return np.asarray(selected.state, dtype=float)
    if hasattr(selected, "I"):
        return np.asarray(selected.I, dtype=float)
    raise RuntimeError("The dataset must contain either 'I' or 'state' for the FFT plot to work.")


def _select_qubit(dataset: xr.Dataset, qubit_name: str) -> xr.Dataset:
    if "qubit" in getattr(dataset, "dims", ()):
        return dataset.sel(qubit=qubit_name)
    return dataset


def add_fit_text(ax, fit):
    """Add fit results text to the axis."""
    fit_frequencies = _fit_values_mhz(fit, "f")
    fit_decays = _fit_values(fit, "decay")
    detuning_correction_mhz = _signed_detuning_correction_mhz(fit)

    t2_us = np.nan
    decay_rate_per_us = np.nan
    if fit_decays.size:
        mean_decay_per_ns = float(np.nanmean(fit_decays))
        if np.isfinite(mean_decay_per_ns) and mean_decay_per_ns > 0:
            t2_us = 1 / mean_decay_per_ns / 1e3
            decay_rate_per_us = mean_decay_per_ns * 1e3

    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Cos fit freq: {_format_detuning_values(fit_frequencies, 'MHz')}",
                f"Fit detuning correction: {detuning_correction_mhz:.3f} MHz",
                f"T2*: {t2_us:.2f} us",
                f"Decay rate: {decay_rate_per_us:.4f} 1/us",
            ]
        ),
        transform=ax.transAxes,
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"),
    )


def _fit_values(fit: xr.Dataset, fit_value: str) -> np.ndarray:
    if "fit_vals" not in fit.dims:
        return np.array([])
    values = fit.sel(fit_vals=fit_value).fit
    return np.asarray(values, dtype=float).reshape(-1)


def _fit_values_mhz(fit: xr.Dataset, fit_value: str) -> dict[int, float]:
    values = {}
    if "detuning_signs" not in fit.dims or "fit_vals" not in fit.dims:
        return values
    for detuning_sign in (1, -1):
        value = float(fit.sel(detuning_signs=detuning_sign, fit_vals=fit_value).fit)
        values[detuning_sign] = abs(value) * 1e3
    return values


def _signed_detuning_correction_mhz(fit: xr.Dataset) -> float:
    if "detuning_signs" not in fit.dims or "fit_vals" not in fit.dims:
        return np.nan
    signed_values = []
    for detuning_sign in (1, -1):
        value = float(fit.sel(detuning_signs=detuning_sign, fit_vals="f").fit)
        signed_values.append(value * detuning_sign)
    return float(np.nanmean(signed_values) * 1e3)


def _format_detuning_values(values: dict[int, float], units: str) -> str:
    if not values:
        return "n/a"
    return ", ".join(f"{sign:+d}: {value:.3f} {units}" for sign, value in values.items())
