"""Analysis and plotting for a constant-envelope Rabi-linearity check."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibration_libs.data import convert_IQ_to_V

from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz


def _oscillation(time_us, amplitude, frequency_mhz, phase, offset):
    return offset + amplitude * np.cos(2 * np.pi * frequency_mhz * time_us + phase)


def fit_rabi_trace(
    time_ns: Sequence[float], signal: Sequence[float]
) -> dict[str, float]:
    """Fit one duration-domain trace and return its oscillation frequency."""
    time_ns = np.asarray(time_ns, dtype=float)
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(time_ns) & np.isfinite(signal)
    time_ns = time_ns[finite]
    signal = signal[finite]
    if time_ns.size < 8 or np.ptp(time_ns) <= 0 or np.ptp(signal) <= 0:
        return {"frequency_hz": np.nan, "r_squared": np.nan}

    order = np.argsort(time_ns)
    time_ns = time_ns[order]
    signal = signal[order]
    time_ns = time_ns - time_ns[0]
    time_us = time_ns / 1000.0
    steps_ns = np.diff(time_ns)
    dt_ns = float(np.median(steps_ns))
    if dt_ns <= 0 or not np.allclose(steps_ns, dt_ns, rtol=0.05, atol=1e-9):
        return {"frequency_hz": np.nan, "r_squared": np.nan}

    centered = signal - np.mean(signal)
    frequencies_hz = np.fft.rfftfreq(time_ns.size, d=dt_ns * 1e-9)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size < 2:
        return {"frequency_hz": np.nan, "r_squared": np.nan}
    spectrum[0] = 0
    initial_frequency_mhz = float(frequencies_hz[int(np.argmax(spectrum))]) / 1e6
    nyquist_mhz = 0.5 / (dt_ns * 1e-3)
    minimum_mhz = 0.25 / np.ptp(time_us)
    scale = float(np.ptp(signal))

    try:
        fitted, _ = curve_fit(
            _oscillation,
            time_us,
            signal,
            p0=(scale / 2, initial_frequency_mhz, 0.0, float(np.mean(signal))),
            bounds=(
                (-2 * scale, minimum_mhz, -4 * np.pi, np.min(signal) - scale),
                (2 * scale, 0.98 * nyquist_mhz, 4 * np.pi, np.max(signal) + scale),
            ),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return {"frequency_hz": np.nan, "r_squared": np.nan}

    fitted_signal = _oscillation(time_us, *fitted)
    residual_sum = float(np.sum((signal - fitted_signal) ** 2))
    total_sum = float(np.sum((signal - np.mean(signal)) ** 2))
    r_squared = np.nan if total_sum == 0 else 1 - residual_sum / total_sum
    return {
        "frequency_hz": abs(float(fitted[1])) * 1e6,
        "r_squared": float(r_squared),
    }


def _fit_variables(
    ds: xr.Dataset, use_state_discrimination: bool
) -> dict[str, xr.DataArray]:
    if use_state_discrimination:
        if "state" not in ds:
            raise RuntimeError("Rabi-linearity dataset is missing state readout.")
        return {"state": ds.state}

    missing = {"I", "Q"} - set(ds.data_vars)
    if missing:
        raise RuntimeError(f"Rabi-linearity dataset is missing {sorted(missing)}.")
    return {"I": ds.I, "Q": ds.Q}


def process_and_fit_dataset(
    ds: xr.Dataset,
    qubits: Sequence[Any],
    *,
    use_state_discrimination: bool,
) -> xr.Dataset:
    """Add predicted and independently measured Rabi frequencies."""
    if not use_state_discrimination:
        ds = convert_IQ_to_V(ds, qubits)
    fit_variables = _fit_variables(ds, use_state_discrimination)

    amplitudes = np.asarray(ds.drive_amplitude_v.values, dtype=float)
    durations = np.asarray(ds.pulse_duration_ns.values, dtype=float)
    measured = np.full((len(qubits), amplitudes.size), np.nan)
    r_squared = np.full_like(measured, np.nan)
    predicted = np.full_like(measured, np.nan)
    selected_channel = np.full(measured.shape, "", dtype="U5")

    for qubit_index, qubit in enumerate(qubits):
        pi_pulse = qubit.xy.operations["x180"]
        predicted[qubit_index] = amplitude_to_rabi_frequency_hz(
            amplitudes,
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        )
        for amplitude_index in range(amplitudes.size):
            candidates = []
            for channel, signal in fit_variables.items():
                trace = signal.isel(
                    qubit=qubit_index, drive_amplitude_v=amplitude_index
                )
                candidates.append((channel, fit_rabi_trace(durations, trace.values)))
            channel, result = max(
                candidates,
                key=lambda item: np.nan_to_num(item[1]["r_squared"], nan=-np.inf),
            )
            measured[qubit_index, amplitude_index] = result["frequency_hz"]
            r_squared[qubit_index, amplitude_index] = result["r_squared"]
            selected_channel[qubit_index, amplitude_index] = channel

    relative_error = np.divide(
        measured - predicted,
        predicted,
        out=np.full_like(measured, np.nan),
        where=predicted != 0,
    )
    ds = ds.assign(
        measured_rabi_frequency_hz=(("qubit", "drive_amplitude_v"), measured),
        predicted_rabi_frequency_hz=(("qubit", "drive_amplitude_v"), predicted),
        rabi_fit_r_squared=(("qubit", "drive_amplitude_v"), r_squared),
        rabi_relative_error=(("qubit", "drive_amplitude_v"), relative_error),
        rabi_fit_channel=(("qubit", "drive_amplitude_v"), selected_channel),
    )
    ds.measured_rabi_frequency_hz.attrs = {
        "long_name": "measured Rabi frequency",
        "units": "Hz",
    }
    ds.predicted_rabi_frequency_hz.attrs = {
        "long_name": "square-pulse-calibration prediction",
        "units": "Hz",
    }
    ds.rabi_relative_error.attrs = {"long_name": "(measured - predicted) / predicted"}
    return ds


def plot_rabi_linearity(ds: xr.Dataset, qubits: Sequence[Any]):
    """Plot raw oscillations, measured frequency, and model residuals."""
    figure, axes = plt.subplots(
        len(qubits),
        3,
        squeeze=False,
        figsize=(15, 4.5 * len(qubits)),
    )
    variable = "state" if "state" in ds else "I"
    for row, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name)
        heatmap = selected[variable].transpose("pulse_duration_ns", "drive_amplitude_v")
        heatmap.plot(
            ax=axes[row, 0], x="drive_amplitude_v", y="pulse_duration_ns", robust=True
        )
        axes[row, 0].set_title(f"{qubit.name}: duration-domain oscillations")
        axes[row, 0].set_xlabel("Constant drive amplitude [V]")
        axes[row, 0].set_ylabel("Pulse duration [ns]")

        amplitude = selected.drive_amplitude_v
        axes[row, 1].plot(
            amplitude,
            selected.predicted_rabi_frequency_hz / 1e6,
            "k--",
            label="square-x180 prediction",
        )
        axes[row, 1].scatter(
            amplitude,
            selected.measured_rabi_frequency_hz / 1e6,
            c=selected.rabi_fit_r_squared,
            vmin=0,
            vmax=1,
            cmap="viridis",
            label="measured",
        )
        axes[row, 1].set_title(f"{qubit.name}: voltage-to-Rabi test")
        axes[row, 1].set_xlabel("Constant drive amplitude [V]")
        axes[row, 1].set_ylabel("Rabi frequency [MHz]")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].legend()

        axes[row, 2].axhline(0, color="black", linewidth=1)
        axes[row, 2].plot(amplitude, 100 * selected.rabi_relative_error, "o-")
        axes[row, 2].set_title(f"{qubit.name}: conversion error")
        axes[row, 2].set_xlabel("Constant drive amplitude [V]")
        axes[row, 2].set_ylabel("Measured − predicted [%]")
        axes[row, 2].grid(alpha=0.25)

    figure.tight_layout()
    return figure
