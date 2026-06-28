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

from profiles import current_profile_name, load_profile
from utils.plotting_settings import (
    FIGURE_SIZE,
    add_calibration_parameter_box,
    qubit_grid_locations,
)
from utils.rabi_amplitude import (
    amplitude_to_rabi_frequency_hz,
    rabi_frequency_hz_to_amplitude,
)

u = unit(coerce_to_integer=True)
MIN_GAUSSIAN_FWHM_R_SQUARED = 0.1
MAX_GAUSSIAN_CENTER_FRACTION_OF_SPAN = 0.5
MAX_GAUSSIAN_FWHM_FRACTION_OF_SPAN = 0.3


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
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if cutoff == 1:
        return [peak_amplitude] * length_ns

    t_cut = length_ns / 2
    tau_ns = t_cut / np.sqrt(1 / cutoff**2 - 1)
    times = np.linspace(-t_cut, t_cut, length_ns)
    envelope = peak_amplitude / np.sqrt(1 + (times / tau_ns) ** 2)
    return envelope.tolist()


def gaussian_envelope(
    length_ns: int,
    cutoff: float,
    peak_amplitude: float,
) -> list[float]:
    """Return a centered Gaussian envelope with edge/peak ratio cutoff."""
    if length_ns < 4:
        raise ValueError("gaussian_length_in_ns must be at least 4 ns.")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")
    if cutoff == 1:
        return [peak_amplitude] * length_ns

    t_cut = length_ns / 2
    sigma_ns = t_cut / np.sqrt(2 * np.log(1 / cutoff))
    times = np.linspace(-t_cut, t_cut, length_ns)
    envelope = peak_amplitude * np.exp(-0.5 * (times / sigma_ns) ** 2)
    return envelope.tolist()


def build_waveform(parameters) -> list[float]:
    """Build the selected Lorentzian-like waveform from calibration parameters."""
    waveform_length = waveform_template_length(parameters)
    if parameters.pulse_shape == "lorentzian":
        waveform = lorentzian_envelope(
            waveform_length,
            parameters.lorentzian_tau_in_ns,
            parameters.lorentzian_peak_amplitude,
        )
    elif parameters.pulse_shape == "root_lorentzian":
        waveform = root_lorentzian_envelope(
            waveform_length,
            parameters.cutoff,
            parameters.lorentzian_peak_amplitude,
        )
    elif parameters.pulse_shape == "gaussian":
        waveform = gaussian_envelope(
            waveform_length,
            parameters.cutoff,
            parameters.lorentzian_peak_amplitude,
        )
    else:
        raise ValueError(
            "pulse_shape must be 'lorentzian', 'root_lorentzian', or 'gaussian'."
        )

    if getattr(parameters, "echo", False):
        waveform = apply_echo_phase_jump(waveform)
    return waveform


def waveform_template_length(parameters) -> int:
    """Return the stored waveform length, which may be shorter than played length."""
    template_length = getattr(parameters, "waveform_template_length_in_ns", None)
    if template_length is None:
        return int(parameters.lorentzian_length_in_ns)

    template_length = int(template_length)
    pulse_length = int(parameters.lorentzian_length_in_ns)
    if template_length < 4:
        raise ValueError("waveform_template_length_in_ns must be at least 4 ns.")
    if template_length > pulse_length:
        raise ValueError(
            "waveform_template_length_in_ns cannot be longer than "
            "lorentzian_length_in_ns."
        )
    return template_length


def lorentzian_play_duration_cycles(parameters) -> int | None:
    """Return QUA play duration cycles when stretching a waveform template."""
    template_length = waveform_template_length(parameters)
    pulse_length = int(parameters.lorentzian_length_in_ns)
    if template_length == pulse_length:
        return None
    if pulse_length % 4 != 0:
        raise ValueError(
            "lorentzian_length_in_ns must be divisible by 4 ns when stretching "
            "a waveform template with QUA duration."
        )
    return pulse_length // 4


def apply_echo_phase_jump(waveform: list[float]) -> list[float]:
    """Flip the sign of the second half of a waveform for an echo pulse."""
    envelope = np.asarray(waveform, dtype=float)
    signs = np.ones_like(envelope)
    signs[len(envelope) // 2 :] = -1
    return (envelope * signs).tolist()


def install_lorentzian_operation(node: QualibrationNode) -> list[float]:
    """Install the selected waveform as a temporary qubit XY operation."""
    waveform = build_waveform(node.parameters)
    sweep_factors = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    if sweep_factors.size == 0:
        raise ValueError("Amplitude sweep is empty.")
    max_factor = float(np.max(np.abs(sweep_factors)))
    max_scaled_amplitude = max(abs(sample) for sample in waveform) * max_factor
    if max_scaled_amplitude >= 0.5:
        raise ValueError(
            "The swept Lorentzian waveform reaches "
            f"{max_scaled_amplitude:g} V. Keep OPX waveform samples below 0.5 V "
            f"(largest played amp_prefactor is {max_factor:g}). Reduce "
            "lorentzian_peak_amplitude or the amplitude sweep range."
        )

    node.namespace["lorentzian_waveform"] = waveform
    node.namespace["lorentzian_play_duration_cycles"] = lorentzian_play_duration_cycles(
        node.parameters
    )
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
    ds = add_gaussian_fwhm_analysis(
        ds,
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    return ds


def add_gaussian_fwhm_analysis(
    ds: xr.Dataset,
    *,
    use_state_discrimination: bool,
) -> xr.Dataset:
    """Fit a Gaussian spectroscopy trace for each amplitude and store FWHM."""
    if not {"qubit", "detuning", "amp_prefactor"}.issubset(ds.coords):
        return ds

    qubits = list(ds.qubit.values)
    amps = list(ds.amp_prefactor.values)
    shape = (len(qubits), len(amps))
    centers = np.full(shape, np.nan, dtype=float)
    fwhm = np.full(shape, np.nan, dtype=float)
    r_squared = np.full(shape, np.nan, dtype=float)

    detuning = np.asarray(ds.detuning.values, dtype=float)
    for qubit_index, qubit_name in enumerate(qubits):
        selected_qubit = ds.sel(qubit=qubit_name)
        for amp_index, amp in enumerate(amps):
            trace = selected_qubit.sel(amp_prefactor=amp)
            signal = _spectroscopy_trace_for_fwhm(
                trace,
                use_state_discrimination=use_state_discrimination,
            )
            center, width, score = _fit_gaussian_center_fwhm(detuning, signal)
            centers[qubit_index, amp_index] = center
            fwhm[qubit_index, amp_index] = width
            r_squared[qubit_index, amp_index] = score

    left = centers - fwhm / 2
    right = centers + fwhm / 2
    ds = ds.assign(
        gaussian_center_hz=(["qubit", "amp_prefactor"], centers),
        gaussian_fwhm_hz=(["qubit", "amp_prefactor"], fwhm),
        gaussian_fwhm_left_hz=(["qubit", "amp_prefactor"], left),
        gaussian_fwhm_right_hz=(["qubit", "amp_prefactor"], right),
        gaussian_fit_r_squared=(["qubit", "amp_prefactor"], r_squared),
    )
    ds.gaussian_center_hz.attrs = {
        "long_name": "Gaussian center detuning",
        "units": "Hz",
    }
    ds.gaussian_fwhm_hz.attrs = {
        "long_name": "Gaussian FWHM",
        "units": "Hz",
    }
    ds.gaussian_fwhm_left_hz.attrs = {
        "long_name": "Gaussian FWHM left edge",
        "units": "Hz",
    }
    ds.gaussian_fwhm_right_hz.attrs = {
        "long_name": "Gaussian FWHM right edge",
        "units": "Hz",
    }
    ds.gaussian_fit_r_squared.attrs = {
        "long_name": "Gaussian fit R squared",
    }
    return ds


def _spectroscopy_trace_for_fwhm(
    trace: xr.Dataset,
    *,
    use_state_discrimination: bool,
) -> np.ndarray:
    if use_state_discrimination:
        return np.asarray(trace["state"].values, dtype=float)
    return np.sqrt(
        np.asarray(trace["I"].values, dtype=float) ** 2
        + np.asarray(trace["Q"].values, dtype=float) ** 2
    )


def _fit_gaussian_center_fwhm(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 5 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan, np.nan, np.nan

    from scipy.optimize import curve_fit

    baseline = float(np.median(y))
    peak_delta = float(np.max(y) - baseline)
    dip_delta = float(baseline - np.min(y))
    is_peak = peak_delta >= dip_delta
    amplitude = peak_delta if is_peak else -dip_delta
    center = float(x[np.argmax(y) if is_peak else np.argmin(y)])
    sigma = float(np.ptp(x) / 6)
    if sigma <= 0:
        return np.nan, np.nan, np.nan

    try:
        fit_offset, fit_amplitude, fit_center, fit_sigma = curve_fit(
            _gaussian,
            x,
            y,
            p0=[baseline, amplitude, center, sigma],
            maxfev=10000,
        )[0]
    except (RuntimeError, ValueError, FloatingPointError):
        return np.nan, np.nan, np.nan

    fit_sigma = abs(float(fit_sigma))
    fit_center = float(fit_center)
    if not np.isfinite(fit_center) or not np.isfinite(fit_sigma) or fit_sigma <= 0:
        return np.nan, np.nan, np.nan

    fitted = _gaussian(x, fit_offset, fit_amplitude, fit_center, fit_sigma)
    residual_sum_squares = float(np.sum((y - fitted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    fwhm = float(2 * np.sqrt(2 * np.log(2)) * fit_sigma)
    max_allowed_center = MAX_GAUSSIAN_CENTER_FRACTION_OF_SPAN * float(np.max(np.abs(x)))

    if (
        not np.isfinite(r_squared)
        or r_squared < MIN_GAUSSIAN_FWHM_R_SQUARED
        or abs(fit_center) > max_allowed_center
        or fwhm > MAX_GAUSSIAN_FWHM_FRACTION_OF_SPAN * np.ptp(x)
    ):
        return np.nan, np.nan, r_squared

    return fit_center, fwhm, r_squared


def _gaussian(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def process_amplitude_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add zero-detuning amplitude coordinates for a 1D Lorentzian sweep."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    full_amp = np.array(
        [
            ds.amp_prefactor * node.parameters.lorentzian_peak_amplitude
            for _ in node.namespace["qubits"]
        ]
    )
    full_freq = np.array([qubit.xy.RF_frequency for qubit in node.namespace["qubits"]])
    ds = ds.assign_coords(
        full_amp=(["qubit", "amp_prefactor"], full_amp),
        full_freq=(["qubit"], full_freq),
    )
    ds.full_amp.attrs = {"long_name": "Lorentzian peak amplitude", "units": "V"}
    ds.full_freq.attrs = {"long_name": "RF frequency at zero detuning", "units": "Hz"}
    ds.attrs.update(_pulse_metadata(node.parameters))
    ds.attrs["detuning_hz"] = 0
    return ds


def _pulse_metadata(parameters) -> dict[str, object]:
    return {
        "operation": getattr(parameters, "operation", None),
        "pulse_shape": getattr(parameters, "pulse_shape", None),
        "lorentzian_length_in_ns": getattr(parameters, "lorentzian_length_in_ns", None),
        "waveform_template_length_in_ns": waveform_template_length(parameters),
        "lorentzian_play_duration_cycles": lorentzian_play_duration_cycles(parameters),
        "lorentzian_tau_in_ns": getattr(parameters, "lorentzian_tau_in_ns", None),
        "cutoff": getattr(parameters, "cutoff", None),
        "lorentzian_peak_amplitude": getattr(
            parameters, "lorentzian_peak_amplitude", None
        ),
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


def plot_amplitude_sweep(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    use_state_discrimination: bool = False,
):
    """Plot a zero-detuning Lorentzian amplitude sweep."""
    variables = ("state",) if use_state_discrimination else ("I", "Q")
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Echo-Lorentzian amplitude plot expected {missing!r} for "
            f"use_state_discrimination={use_state_discrimination}, "
            f"but dataset contains {list(ds.data_vars)}"
        )

    figure, axes = plt.subplots(
        len(qubits) * len(variables),
        1,
        squeeze=False,
        figsize=FIGURE_SIZE,
    )
    axis_index = 0
    for qubit in qubits:
        selected = _with_amplitude_plot_coords(ds.sel(qubit=qubit.name), qubit)
        for variable in variables:
            ax = axes[axis_index, 0]
            scale = 1 if variable == "state" else 1e3
            ylabel = "Measured state" if variable == "state" else f"{variable} [mV]"
            (selected[variable] * scale).plot(
                ax=ax,
                x="rabi_frequency_MHz",
                marker="o",
                linewidth=1.2,
            )
            _add_absolute_amplitude_xaxis(ax, qubit)
            ax.set_title(f"{qubit.name}: {variable} at zero detuning")
            ax.set_xlabel("Rabi frequency [MHz]")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            axis_index += 1

    _finish_figure_layout(
        figure,
        "Echo Lorentzian amplitude sweep: zero detuning",
        ds,
        qubits,
    )
    return figure


def _with_plot_coords(selected: xr.Dataset, qubit: AnyTransmon) -> xr.Dataset:
    pi_pulse = qubit.xy.operations["x180"]
    return selected.assign_coords(
        detuning_MHz=selected.detuning / u.MHz,
        rabi_frequency_MHz=amplitude_to_rabi_frequency_hz(
            selected.full_amp,
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        )
        / u.MHz,
    )


def _with_amplitude_plot_coords(selected: xr.Dataset, qubit: AnyTransmon) -> xr.Dataset:
    pi_pulse = qubit.xy.operations["x180"]
    return selected.assign_coords(
        rabi_frequency_MHz=amplitude_to_rabi_frequency_hz(
            selected.full_amp,
            float(pi_pulse.amplitude),
            float(pi_pulse.length),
        )
        / u.MHz,
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


def _add_absolute_amplitude_xaxis(ax, qubit: AnyTransmon) -> None:
    pi_pulse = qubit.xy.operations["x180"]
    pi_amp = float(pi_pulse.amplitude)
    pi_length_ns = float(pi_pulse.length)

    def rabi_mhz_to_amp_v(rabi_mhz):
        return rabi_frequency_hz_to_amplitude(rabi_mhz * u.MHz, pi_amp, pi_length_ns)

    def amp_v_to_rabi_mhz(amp_v):
        return amplitude_to_rabi_frequency_hz(amp_v, pi_amp, pi_length_ns) / u.MHz

    top_axis = ax.secondary_xaxis(
        "top",
        functions=(rabi_mhz_to_amp_v, amp_v_to_rabi_mhz),
    )
    top_axis.set_xlabel("Lorentzian peak amplitude [V]")


def _add_absolute_amplitude_axis(ax, qubit: AnyTransmon) -> None:
    pi_pulse = qubit.xy.operations["x180"]
    pi_amp = float(pi_pulse.amplitude)
    pi_length_ns = float(pi_pulse.length)

    def rabi_mhz_to_amp_v(rabi_mhz):
        return rabi_frequency_hz_to_amplitude(rabi_mhz * u.MHz, pi_amp, pi_length_ns)

    def amp_v_to_rabi_mhz(amp_v):
        return amplitude_to_rabi_frequency_hz(amp_v, pi_amp, pi_length_ns) / u.MHz

    right_axis = ax.secondary_yaxis(
        "right",
        functions=(rabi_mhz_to_amp_v, amp_v_to_rabi_mhz),
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


def _format_mv(value) -> str | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return f"{1e3 * value:.3f} mV"


def _parameter_lines(ds: xr.Dataset, qubits: List[AnyTransmon]) -> list[str]:
    lines = ["Parameters"]

    pulse_parts = [
        f"pulse shape={ds.attrs.get('pulse_shape')}",
        f"echo={bool(ds.attrs.get('echo', False))}",
        f"pulse length={_format_value(ds.attrs.get('lorentzian_length_in_ns'), ' ns')}",
    ]
    template_length = ds.attrs.get("waveform_template_length_in_ns")
    pulse_length = ds.attrs.get("lorentzian_length_in_ns")
    if template_length is not None and template_length != pulse_length:
        pulse_parts.append(f"template length={_format_value(template_length, ' ns')}")
        duration_cycles = ds.attrs.get("lorentzian_play_duration_cycles")
        if duration_cycles is not None:
            pulse_parts.append(
                f"play duration={_format_value(duration_cycles, ' cycles')}"
            )
    lines.append(", ".join(part for part in pulse_parts if part))

    shape_parts = [
        f"peak amp={_format_mv(ds.attrs.get('lorentzian_peak_amplitude'))}",
    ]
    if ds.attrs.get("pulse_shape") in {"root_lorentzian", "gaussian"}:
        shape_parts.append(_format_value(ds.attrs.get("cutoff"), " cutoff"))
    else:
        shape_parts.append(
            _format_value(ds.attrs.get("lorentzian_tau_in_ns"), " ns tau")
        )
    lines.append(", ".join(part for part in shape_parts if part))

    sweep_parts = [
        f"amp factor={_format_value(ds.attrs.get('min_amp_factor'))}:"
        f"{_format_value(ds.attrs.get('amp_factor_step'))}:"
        f"{_format_value(ds.attrs.get('max_amp_factor'))}",
    ]
    span = _format_value(ds.attrs.get("frequency_span_in_mhz"), " MHz")
    step = _format_value(ds.attrs.get("frequency_step_in_mhz"), " MHz")
    if span is not None and step is not None:
        sweep_parts.append(f"detuning span={span}, step={step}")
    lines.append(", ".join(part for part in sweep_parts if part))

    for qubit in qubits:
        qubit_parts = []
        rf_frequency = getattr(getattr(qubit, "xy", None), "RF_frequency", None)
        if rf_frequency is not None:
            qubit_parts.append(
                f"current drive f01={float(rf_frequency) / u.GHz:.6f} GHz"
            )

        pi_pulse = qubit.xy.operations["x180"]
        qubit_parts.append(
            f"x180 square pi: amp={_format_mv(float(pi_pulse.amplitude))}, "
            f"t_pi={float(pi_pulse.length):g} ns"
        )

        coherence_parts = [
            _format_seconds(_t1_seconds(qubit), "T1"),
            _format_seconds(_t2_seconds(qubit), "T2"),
            _format_hz(_t2_limit_hz(qubit), "1/(pi*T2)"),
        ]
        qubit_parts.extend(part for part in coherence_parts if part)
        if qubit_parts:
            lines.append(f"{qubit.name}: " + " | ".join(qubit_parts))

    return [line for line in lines if line]


def _t1_seconds(qubit: AnyTransmon) -> float | None:
    profile_value = _profile_coherence_seconds(qubit.name, "t1_ns")
    if profile_value is not None:
        return profile_value

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


def _profile_coherence_seconds(qubit_name: str, metric_name: str) -> float | None:
    try:
        profile = load_profile(current_profile_name())
        value = profile["metrics"]["qubits"][qubit_name]["coherence"].get(metric_name)
    except Exception:
        return None
    return _coherence_value_to_seconds(value)


def _coherence_value_to_seconds(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return None
    # Some metrics fields still carry a "_ns" name even after seconds-valued
    # calibration updates. Small values are treated as already in seconds.
    return value if value < 1e-3 else value * 1e-9


def _finish_figure_layout(
    figure, title: str, ds: xr.Dataset, qubits: List[AnyTransmon]
) -> None:
    pulse_name = ds.attrs.get("pulse_shape") or ds.attrs.get("operation")
    figure_title = f"{title} - {pulse_name}" if pulse_name else title
    figure.suptitle(figure_title, y=0.99)
    parameter_lines = _parameter_lines(ds, qubits)
    add_calibration_parameter_box(figure, parameter_lines, gid="lorentzian_parameters")
    bottom = min(0.25, 0.055 + 0.018 * len(parameter_lines))
    figure.subplots_adjust(top=0.9, bottom=bottom, right=0.86, hspace=0.35, wspace=0.45)


def _t2_seconds(qubit: AnyTransmon) -> float | None:
    for metric_name in ("t2_ramsey", "t2_ramsey_ns", "t2_echo", "t2_echo_ns"):
        profile_value = _profile_coherence_seconds(qubit.name, metric_name)
        if profile_value is not None:
            return profile_value

    for attribute in (
        "t2_ramsey",
        "t2_ramsey_ns",
        "T2ramsey",
        "t2_echo",
        "t2_echo_ns",
        "T2echo",
    ):
        value = getattr(qubit, attribute, None)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            continue
        return _coherence_value_to_seconds(value)
    return None


def _add_t2_limit_lines(ax, qubit: AnyTransmon) -> None:
    t2_s = _t2_seconds(qubit)
    if t2_s is None:
        return

    limit_mhz = 1 / (2 * np.pi * t2_s) / u.MHz
    for index, detuning_mhz in enumerate((-limit_mhz, limit_mhz)):
        ax.axvline(
            detuning_mhz,
            color="white",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="T2 limit: ±1/(2πT2)" if index == 0 else None,
        )


def _add_fwhm_markers(ax, selected: xr.Dataset) -> None:
    required = {"gaussian_fwhm_left_hz", "gaussian_fwhm_right_hz", "rabi_frequency_MHz"}
    if not required.issubset(set(selected.variables)):
        return

    y = np.asarray(selected.rabi_frequency_MHz.values, dtype=float)
    left = np.asarray(selected.gaussian_fwhm_left_hz.values, dtype=float) / u.MHz
    right = np.asarray(selected.gaussian_fwhm_right_hz.values, dtype=float) / u.MHz
    for x_values, marker in ((left, "<"), (right, ">")):
        finite = np.isfinite(x_values) & np.isfinite(y)
        if not np.any(finite):
            continue
        ax.scatter(
            x_values[finite],
            y[finite],
            marker=marker,
            s=28,
            facecolors="none",
            edgecolors="red",
            linewidths=1.1,
            label="Gaussian FWHM" if marker == "<" else None,
            zorder=5,
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def _plot_state(ds: xr.Dataset, qubits: List[AnyTransmon]):
    qubits_by_name = {qubit.name: qubit for qubit in qubits}
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit_ref in grid_iter(grid):
        qubit_name = qubit_ref["qubit"]
        qubit = qubits_by_name[qubit_name]
        selected = _with_plot_coords(ds.sel(qubit=qubit_name), qubit)
        plotted = (
            selected["state"]
            .transpose("amp_prefactor", "detuning")
            .plot(
                ax=ax,
                x="detuning_MHz",
                y="rabi_frequency_MHz",
                add_colorbar=True,
                cbar_kwargs={"pad": 0.16},
            )
        )
        plotted.colorbar.set_label("Measured state")
        _add_absolute_frequency_axis(ax, _rf_frequency_ghz(selected))
        _add_absolute_amplitude_axis(ax, qubit)
        _add_t2_limit_lines(ax, qubit)
        _add_fwhm_markers(ax, selected)
        ax.set_title(f"{qubit_name}: measured state")
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Rabi frequency [MHz]")

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
            plotted = (
                selected[variable].transpose("amp_prefactor", "detuning") / u.mV
            ).plot(
                ax=ax,
                x="detuning_MHz",
                y="rabi_frequency_MHz",
                add_colorbar=True,
                robust=True,
                cbar_kwargs={"pad": 0.16},
            )
            label = f"{variable} [mV]"
            plotted.colorbar.set_label(label)
            _add_absolute_frequency_axis(ax, rf_frequency_ghz)
            _add_absolute_amplitude_axis(ax, qubit)
            _add_t2_limit_lines(ax, qubit)
            _add_fwhm_markers(ax, selected)
            ax.set_title(f"{qubit_name}: {label}")
            ax.set_xlabel("Detuning [MHz]")
            ax.set_ylabel("Rabi frequency [MHz]")

    _finish_figure_layout(figure, "Echo Lorentzian: I and Q quadratures", ds, qubits)
    return figure


if __name__ == "__main__":
    # Example demonstration: plot Lorentzian, root-Lorentzian (echo) and Gaussian
    import matplotlib.pyplot as _plt

    LENGTH = 201
    TAU = 20.0
    PEAK = 1
    CUTOFF = 0.25

    lor = lorentzian_envelope(LENGTH, TAU, PEAK)
    root_lor = root_lorentzian_envelope(LENGTH, CUTOFF, PEAK)
    gauss = gaussian_envelope(LENGTH, CUTOFF, PEAK)
    echo_lor = apply_echo_phase_jump(root_lor)

    times = np.arange(LENGTH) - (LENGTH - 1) / 2

    fig, ax = _plt.subplots()
    ax.axhline(CUTOFF, color="gray", linestyle=":", label="Lorentzian tau")
    ax.plot(times, lor, label="Lorentzian")
    ax.plot(times, root_lor, label=f"Root-Lorentzian cutoff={CUTOFF}")
    ax.plot(times, gauss, label=f"Gaussian cutoff={CUTOFF}")
    ax.plot(times, echo_lor, label="Echo Lorentzian (phase-jump)", linestyle="--")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Example pulse shapes")
    ax.legend()
    fig.tight_layout()
