from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.qubit_spectroscopy.analysis import MIN_FIT_R_SQUARED
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import (
    FIGURE_SIZE,
    CalibrationPlot,
    add_calibration_parameter_box,
)

u = unit(coerce_to_integer=True)


def _add_detuning_axis(ax, current_frequency_ghz: float):
    """Add a top x-axis showing detuning from the configured resonance."""
    detuning_axis = ax.secondary_xaxis(
        "top",
        functions=(
            lambda frequency_ghz: (frequency_ghz - current_frequency_ghz) * 1e3,
            lambda detuning_mhz: current_frequency_ghz + detuning_mhz / 1e3,
        ),
    )
    detuning_axis.set_xlabel("Detuning from current resonance [MHz]")
    return detuning_axis


def _transition_frequency(qubit, transition: str) -> float:
    if transition == "ef":
        if qubit.f_12 is not None:
            return float(qubit.f_12)
        return float(qubit.f_01 - qubit.anharmonicity)
    return float(qubit.xy.RF_frequency)


def _lorentzian_peak(x, offset, amplitude, center, gamma):
    return offset + amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)


def _fit_value(fit: xr.Dataset, name: str) -> float:
    return float(fit[name].values)


def _can_plot_lorentzian_fit(fit: xr.Dataset) -> bool:
    required = {
        "fit_offset",
        "fit_amplitude",
        "fit_position",
        "fit_gamma",
        "fit_r_squared",
    }
    if not required.issubset(set(fit.variables)):
        return False
    values = [_fit_value(fit, name) for name in required]
    return all(np.isfinite(value) for value in values) and (
        _fit_value(fit, "fit_r_squared") >= MIN_FIT_R_SQUARED
    )


def _is_selected_fit_trace(fit: xr.Dataset, variable: str) -> bool:
    if variable == "state":
        return True
    if "selected_quadrature" not in fit:
        return False
    return str(fit.selected_quadrature.values) == variable


def _plot_lorentzian_fit(ax, trace: xr.Dataset, fit: xr.Dataset, scale: float) -> None:
    x = np.asarray(trace.detuning.values, dtype=float)
    if x.size == 0:
        return
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 500)
    y_dense = _lorentzian_peak(
        x_dense,
        _fit_value(fit, "fit_offset"),
        _fit_value(fit, "fit_amplitude"),
        _fit_value(fit, "fit_position"),
        _fit_value(fit, "fit_gamma"),
    )
    current_frequency_hz = (
        np.asarray(trace.full_freq_GHz.values, dtype=float)[0] * u.GHz - x[0]
    )
    ax.plot(
        (current_frequency_hz + x_dense) / u.GHz,
        y_dense * scale,
        color="tab:red",
        linewidth=1.4,
        label=f"Lorentzian fit R^2={_fit_value(fit, 'fit_r_squared'):.3f}",
    )


def _plot_measured_max(ax, fit: xr.Dataset, current_frequency_ghz: float) -> None:
    if "measured_max_position" not in fit:
        return
    measured_position = _fit_value(fit, "measured_max_position")
    if not np.isfinite(measured_position):
        return
    ax.axvline(
        current_frequency_ghz + measured_position / u.GHz,
        color="tab:gray",
        linestyle=":",
        label=(
            "Measured max: "
            f"{current_frequency_ghz + measured_position / u.GHz:.6f} GHz"
        ),
    )


def _spectroscopy_parameter_lines(
    qubits,
    fits: xr.Dataset,
    operation: str,
    operation_amplitude_factor: float,
    operation_len_in_ns: Optional[int],
    transition: str,
):
    """Describe the pulse settings and configured/fitted frequencies."""
    lines = [
        "Parameters",
        f"operation={operation}, amplitude factor={operation_amplitude_factor:g}",
    ]
    for qubit in qubits:
        pulse = getattr(qubit.xy, "operations", {}).get(operation)
        configured_amplitude = getattr(pulse, "amplitude", None)
        configured_length = getattr(pulse, "length", None)
        played_length = operation_len_in_ns if operation_len_in_ns is not None else configured_length
        fitted_frequency_ghz = float(fits.sel(qubit=qubit.name).res_freq.values) / u.GHz
        current_f01_ghz = float(qubit.xy.RF_frequency) / u.GHz

        pulse_parts = [f"{qubit.name}: current drive f01={current_f01_ghz:.6f} GHz"]
        if transition == "ef":
            pulse_parts.append(f"fitted/new ef={fitted_frequency_ghz:.6f} GHz")
        else:
            pulse_parts.append(f"fitted/new f01={fitted_frequency_ghz:.6f} GHz")
        if played_length is not None:
            pulse_parts.append(f"pulse length={float(played_length):g} ns")
        if configured_amplitude is not None:
            played_amplitude = float(configured_amplitude) * operation_amplitude_factor
            pulse_parts.append(
                f"pulse amp={1e3 * played_amplitude:.3f} mV "
                f"(configured {1e3 * float(configured_amplitude):.3f} mV)"
            )
        lines.append(" | ".join(pulse_parts))
    return lines


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    use_state_discrimination: bool = False,
    transition: str = "ge",
    operation: str = "saturation",
    operation_amplitude_factor: float = 1.0,
    operation_len_in_ns: Optional[int] = None,
):
    """
    Plot the raw I and Q qubit-spectroscopy responses on separate subplots.

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
    - The fitted qubit frequency is marked on both subplots.
    """
    variables = ["state"] if use_state_discrimination else ["I", "Q"]
    missing = [variable for variable in variables if variable not in ds]
    if missing:
        raise RuntimeError(
            f"Qubit-spectroscopy plot expected {variables} for "
            f"use_state_discrimination={use_state_discrimination}, but dataset contains {list(ds.data_vars)}"
        )

    fig, axes = plt.subplots(
        len(variables) * len(qubits),
        1,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )

    for qubit_index, qubit in enumerate(qubits):
        selected = ds.sel(qubit=qubit.name).assign_coords(
            full_freq_GHz=ds.full_freq.sel(qubit=qubit.name) / u.GHz
        )
        fit = fits.sel(qubit=qubit.name)
        fitted_frequency_ghz = float(fit.res_freq.values) / u.GHz
        current_frequency_ghz = _transition_frequency(qubit, transition) / u.GHz
        current_ge_frequency_ghz = float(qubit.f_01) / u.GHz if transition == "ef" else None
        sweep_limits = (
            float(selected.full_freq_GHz.min()),
            float(selected.full_freq_GHz.max()),
        )

        start = len(variables) * qubit_index
        qubit_axes = axes[start : start + len(variables), 0]
        colors = ("tab:blue", "tab:orange", "tab:green")
        for ax, variable, color in zip(qubit_axes, variables, colors):
            scale = 1 if variable == "state" else 1 / u.mV
            label = (
                "Measured state"
                if variable == "state"
                else "Rotated I [mV]"
                if variable == "I_rot"
                else f"{variable} [mV]"
            )
            trace = selected if variable in selected else fit.assign_coords(
                full_freq_GHz=selected.full_freq_GHz
            )
            (trace[variable] * scale).plot(ax=ax, x="full_freq_GHz", color=color)
            ax.axvline(
                current_frequency_ghz,
                color="black",
                linestyle=":",
                label=(
                    f"Current drive f01: {current_frequency_ghz:.6f} GHz"
                    if transition == "ge"
                    else f"Current ef: {current_frequency_ghz:.6f} GHz"
                ),
            )
            if current_ge_frequency_ghz is not None:
                ax.axvline(
                    current_ge_frequency_ghz,
                    color="tab:purple",
                    linestyle=":",
                    label=f"Current drive f01: {current_ge_frequency_ghz:.6f} GHz",
                )
            ax.axvline(
                fitted_frequency_ghz,
                color="tab:red",
                linestyle="--",
                label=(
                    f"Fitted new f01: {fitted_frequency_ghz:.6f} GHz"
                    if transition == "ge"
                    else f"Fitted new ef: {fitted_frequency_ghz:.6f} GHz"
                ),
            )
            _plot_measured_max(ax, fit, current_frequency_ghz)
            if _is_selected_fit_trace(fit, variable) and _can_plot_lorentzian_fit(fit):
                _plot_lorentzian_fit(ax, trace, fit, scale)
            ax.set_xlim(*sweep_limits)
            ax.set_title(f"{qubit.name}: {label}")
            ax.set_xlabel("RF frequency [GHz]")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.25)
            _add_detuning_axis(ax, current_frequency_ghz)

    fig.suptitle(
        "Qubit spectroscopy: measured state"
        if use_state_discrimination
        else "Qubit spectroscopy: I and Q quadratures"
    )
    parameter_lines = _spectroscopy_parameter_lines(
        qubits,
        fits,
        operation,
        operation_amplitude_factor,
        operation_len_in_ns,
        transition,
    )
    add_calibration_parameter_box(fig, parameter_lines, gid="spectroscopy_parameters")
    calibration_plot = CalibrationPlot(fig)
    calibration_plot.add_timestamp()
    calibration_plot.tight_layout_for_parameters(len(parameter_lines))
    return fig
