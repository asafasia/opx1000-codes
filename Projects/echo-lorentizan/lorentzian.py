from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam.components.pulses import WaveformPulse
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations
from utils.rabi_amplitude import (
    amplitude_to_rabi_frequency_hz,
    rabi_frequency_hz_to_amplitude,
)


u = unit(coerce_to_integer=True)


def lorentzian_envelope(
    length_ns: int,
    tau_ns: float,
    peak_amplitude: float,
) -> list[float]:
    """Return a centered Lorentzian envelope A / (1 + (t / tau)^2)."""
    if length_ns < 4:
        raise ValueError("lorentzian_length_in_ns must be at least 4 ns.")
    if tau_ns <= 0:
        raise ValueError("lorentzian_tau_in_ns must be positive.")

    times = np.arange(length_ns, dtype=float) - (length_ns - 1) / 2
    envelope = peak_amplitude / (1 + (times / tau_ns) ** 2)
    return envelope.tolist()


def root_lorentzian_envelope(
    length_ns: int,
    cutoff: float,
    peak_amplitude: float,
) -> list[float]:
    """Return a centered root-Lorentzian envelope with edge/peak ratio cutoff."""
    if length_ns < 4:
        raise ValueError("root_lorentzian_length_in_ns must be at least 4 ns.")
    if not 0 < cutoff <= 1:
        raise ValueError("root_lorentzian_cutoff must satisfy 0 < cutoff <= 1.")
    if cutoff == 1:
        return [peak_amplitude] * length_ns

    t_cut = length_ns / 2
    tau_ns = t_cut / np.sqrt(1 / cutoff**2 - 1)
    times = np.linspace(-t_cut, t_cut, length_ns)
    envelope = peak_amplitude / np.sqrt(1 + (times / tau_ns) ** 2)
    return envelope.tolist()


def build_waveform(parameters) -> list[float]:
    """Build the selected Lorentzian-like waveform from calibration parameters."""
    if parameters.pulse_shape == "lorentzian":
        waveform = lorentzian_envelope(
            parameters.lorentzian_length_in_ns,
            parameters.lorentzian_tau_in_ns,
            parameters.lorentzian_peak_amplitude,
        )
    elif parameters.pulse_shape == "root_lorentzian":
        waveform = root_lorentzian_envelope(
            parameters.lorentzian_length_in_ns,
            parameters.root_lorentzian_cutoff,
            parameters.lorentzian_peak_amplitude,
        )
    else:
        raise ValueError("pulse_shape must be either 'lorentzian' or 'root_lorentzian'.")

    if getattr(parameters, "echo", False):
        waveform = apply_echo_phase_jump(waveform)
    return waveform


def apply_echo_phase_jump(waveform: list[float]) -> list[float]:
    """Flip the sign of the second half of a waveform for an echo pulse."""
    envelope = np.asarray(waveform, dtype=float)
    signs = np.ones_like(envelope)
    signs[len(envelope) // 2 :] = -1
    return (envelope * signs).tolist()


def install_lorentzian_operation(node: QualibrationNode) -> list[float]:
    """Install the selected waveform as a temporary qubit XY operation."""
    waveform = build_waveform(node.parameters)
    max_scaled_amplitude = max(abs(sample) for sample in waveform) * max(
        abs(node.parameters.min_amp_factor),
        abs(node.parameters.max_amp_factor),
    )
    if max_scaled_amplitude >= 0.5:
        raise ValueError(
            "The swept Lorentzian waveform reaches "
            f"{max_scaled_amplitude:g} V. Keep OPX waveform samples below 0.5 V "
            "by reducing lorentzian_peak_amplitude or max_amp_factor."
        )

    node.namespace["lorentzian_waveform"] = waveform
    for qubit in node.namespace["qubits"]:
        qubit.xy.operations[node.parameters.operation] = WaveformPulse(
            waveform_I=waveform,
            waveform_Q=[0.0] * len(waveform),
        )
    return waveform


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add physical frequency and Lorentzian peak-amplitude coordinates."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    full_freq = np.array(
        [ds.detuning + qubit.xy.RF_frequency for qubit in node.namespace["qubits"]]
    )
    full_amp = np.array(
        [
            ds.amp_prefactor * node.parameters.lorentzian_peak_amplitude
            for _ in node.namespace["qubits"]
        ]
    )
    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq),
        full_amp=(["qubit", "amp_prefactor"], full_amp),
    )
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    ds.full_amp.attrs = {"long_name": "Lorentzian peak amplitude", "units": "V"}
    ds.attrs.update(_pulse_metadata(node.parameters))
    return ds


def _pulse_metadata(parameters) -> dict[str, object]:
    return {
        "pulse_shape": getattr(parameters, "pulse_shape", None),
        "lorentzian_length_in_ns": getattr(parameters, "lorentzian_length_in_ns", None),
        "lorentzian_tau_in_ns": getattr(parameters, "lorentzian_tau_in_ns", None),
        "root_lorentzian_cutoff": getattr(parameters, "root_lorentzian_cutoff", None),
        "lorentzian_peak_amplitude": getattr(parameters, "lorentzian_peak_amplitude", None),
        "echo": getattr(parameters, "echo", False),
        "min_amp_factor": getattr(parameters, "min_amp_factor", None),
        "max_amp_factor": getattr(parameters, "max_amp_factor", None),
        "amp_factor_step": getattr(parameters, "amp_factor_step", None),
        "frequency_span_in_mhz": getattr(parameters, "frequency_span_in_mhz", None),
        "frequency_step_in_mhz": getattr(parameters, "frequency_step_in_mhz", None),
    }


def plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
):
    """Plot Lorentzian data with detuning below and absolute RF frequency above."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Echo-Lorentzian plot expected {missing!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    return _plot_state(ds, qubits) if use_state_discrimination else _plot_iq(ds, qubits)


def _with_plot_coords(selected: xr.Dataset, qubit: AnyTransmon) -> xr.Dataset:
    pi_pulse = qubit.xy.operations["x180"]
    return selected.assign_coords(
        detuning_MHz=selected.detuning / u.MHz,
        rabi_frequency_Hz=amplitude_to_rabi_frequency_hz(
            selected.full_amp,
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        ),
    )


def _rf_frequency_ghz(selected: xr.Dataset) -> float:
    return float(((selected.full_freq - selected.detuning).isel(detuning=0)) / u.GHz)


def _add_absolute_frequency_axis(ax, rf_frequency_ghz: float) -> None:
    def detuning_mhz_to_rf_ghz(detuning_mhz):
        return rf_frequency_ghz + detuning_mhz / 1000

    def rf_ghz_to_detuning_mhz(rf_ghz):
        return (rf_ghz - rf_frequency_ghz) * 1000

    top_axis = ax.secondary_xaxis(
        "top",
        functions=(detuning_mhz_to_rf_ghz, rf_ghz_to_detuning_mhz),
    )
    top_axis.set_xlabel("RF frequency [GHz]")


def _add_absolute_amplitude_axis(ax, qubit: AnyTransmon) -> None:
    pi_pulse = qubit.xy.operations["x180"]
    pi_amp = float(pi_pulse.amplitude)
    pi_length_ns = float(pi_pulse.length)

    def rabi_hz_to_amp_v(rabi_hz):
        return rabi_frequency_hz_to_amplitude(rabi_hz, pi_amp, pi_length_ns)

    def amp_v_to_rabi_hz(amp_v):
        return amplitude_to_rabi_frequency_hz(amp_v, pi_amp, pi_length_ns)

    right_axis = ax.secondary_yaxis(
        "right",
        functions=(rabi_hz_to_amp_v, amp_v_to_rabi_hz),
    )
    right_axis.set_ylabel("Lorentzian peak amplitude [V]")


def _format_value(value, suffix: str = "") -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _format_seconds(seconds: float | None, label: str) -> str | None:
    if seconds is None:
        return None
    return f"{label}={seconds * 1e6:g} us"


def _format_hz(hz: float | None, label: str) -> str | None:
    if hz is None:
        return None
    return f"{label}={hz:g} Hz"


def _parameter_summary(ds: xr.Dataset, qubits: List[AnyTransmon]) -> str | None:
    parts = [
        _format_value(ds.attrs.get("pulse_shape")),
        _format_value(ds.attrs.get("lorentzian_length_in_ns"), " ns"),
        f"echo={bool(ds.attrs.get('echo', False))}",
        _format_value(ds.attrs.get("lorentzian_peak_amplitude"), " V peak"),
    ]
    if ds.attrs.get("pulse_shape") == "root_lorentzian":
        parts.append(_format_value(ds.attrs.get("root_lorentzian_cutoff"), " cutoff"))
    else:
        parts.append(_format_value(ds.attrs.get("lorentzian_tau_in_ns"), " ns tau"))

    span = _format_value(ds.attrs.get("frequency_span_in_mhz"), " MHz span")
    step = _format_value(ds.attrs.get("frequency_step_in_mhz"), " MHz step")
    if span and step:
        parts.append(f"{span}, {step}")

    coherence_summary = _coherence_summary(qubits)
    if coherence_summary:
        parts.append(coherence_summary)

    if len(qubits) == 1:
        pi_pulse = qubits[0].xy.operations["x180"]
        parts.append(
            f"x180 square pi: amp={float(pi_pulse.amplitude):g} V, "
            f"t_pi={float(pi_pulse.length):g} ns"
        )

    present_parts = [part for part in parts if part]
    return " | ".join(present_parts) if present_parts else None


def _t1_seconds(qubit: AnyTransmon) -> float | None:
    for attribute in ("T1", "t1_ns"):
        value = getattr(qubit, attribute, None)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            continue
        return value * 1e-9 if attribute.endswith("_ns") else value
    return None


def _t2_limit_hz(qubit: AnyTransmon) -> float | None:
    t2_s = _t2_seconds(qubit)
    return None if t2_s is None else 1 / (np.pi * t2_s)


def _coherence_summary(qubits: List[AnyTransmon]) -> str | None:
    qubit_parts = []
    for qubit in qubits:
        values = [
            _format_seconds(_t1_seconds(qubit), "T1"),
            _format_seconds(_t2_seconds(qubit), "T2"),
            _format_hz(_t2_limit_hz(qubit), "1/(pi*T2)"),
        ]
        present_values = [value for value in values if value]
        if present_values:
            qubit_parts.append(f"{qubit.name}: " + ", ".join(present_values))
    return "; ".join(qubit_parts) if qubit_parts else None


def _finish_figure_layout(figure, title: str, ds: xr.Dataset, qubits: List[AnyTransmon]) -> None:
    figure.suptitle(title, y=0.99)
    summary = _parameter_summary(ds, qubits)
    top = 0.9
    if summary:
        figure.text(
            0.5,
            0.945,
            summary,
            ha="center",
            va="top",
            fontsize=9,
        )
        top = 0.84
    figure.subplots_adjust(top=top, right=0.86, hspace=0.35, wspace=0.45)


def _t2_seconds(qubit: AnyTransmon) -> float | None:
    for attribute in ("T2ramsey", "T2echo", "t2_ramsey_ns", "t2_echo_ns"):
        value = getattr(qubit, attribute, None)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            continue
        return value * 1e-9 if attribute.endswith("_ns") else value
    return None


def _add_t2_limit_lines(ax, qubit: AnyTransmon) -> None:
    t2_s = _t2_seconds(qubit)
    if t2_s is None:
        return

    limit_mhz = 1 / (2 * np.pi * t2_s) / u.MHz
    for detuning_mhz in (-limit_mhz, limit_mhz):
        ax.axvline(
            detuning_mhz,
            color="white",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )


def _plot_state(ds: xr.Dataset, qubits: List[AnyTransmon]):
    qubits_by_name = {qubit.name: qubit for qubit in qubits}
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit_ref in grid_iter(grid):
        qubit_name = qubit_ref["qubit"]
        qubit = qubits_by_name[qubit_name]
        selected = _with_plot_coords(ds.sel(qubit=qubit_name), qubit)
        plotted = selected["state"].plot(
            ax=ax,
            x="detuning_MHz",
            y="rabi_frequency_Hz",
            add_colorbar=True,
            robust=True,
            cbar_kwargs={"pad": 0.16},
        )
        plotted.colorbar.set_label("Measured state")
        _add_absolute_frequency_axis(ax, _rf_frequency_ghz(selected))
        _add_absolute_amplitude_axis(ax, qubit)
        _add_t2_limit_lines(ax, qubit)
        ax.set_title(f"{qubit_name}: measured state")
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Rabi frequency [Hz]")

    grid.fig.set_size_inches(*FIGURE_SIZE)
    _finish_figure_layout(grid.fig, "Echo Lorentzian: state", ds, qubits)
    return grid.fig


def _plot_iq(ds: xr.Dataset, qubits: List[AnyTransmon]):
    num_qubits = len(qubits)
    figure, axes = plt.subplots(
        num_qubits,
        2,
        squeeze=False,
        figsize=(FIGURE_SIZE[0] * 2, FIGURE_SIZE[1] * max(1, num_qubits)),
    )

    for row, qubit in enumerate(qubits):
        qubit_name = qubit.name
        selected = _with_plot_coords(ds.sel(qubit=qubit_name), qubit)
        rf_frequency_ghz = _rf_frequency_ghz(selected)
        for column, variable in enumerate(("I", "Q")):
            ax = axes[row][column]
            plotted = (selected[variable] / u.mV).plot(
                ax=ax,
                x="detuning_MHz",
                y="rabi_frequency_Hz",
                add_colorbar=True,
                robust=True,
                cbar_kwargs={"pad": 0.16},
            )
            label = f"{variable} [mV]"
            plotted.colorbar.set_label(label)
            _add_absolute_frequency_axis(ax, rf_frequency_ghz)
            _add_absolute_amplitude_axis(ax, qubit)
            _add_t2_limit_lines(ax, qubit)
            ax.set_title(f"{qubit_name}: {label}")
            ax.set_xlabel("Detuning [MHz]")
            ax.set_ylabel("Rabi frequency [Hz]")

    _finish_figure_layout(figure, "Echo Lorentzian: I and Q quadratures", ds, qubits)
    return figure
