from typing import List, Optional
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

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
    variables = ("state",) if use_state_discrimination else ("I", "Q")
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
        for ax, variable, color in zip(qubit_axes, variables, ("tab:blue", "tab:orange")):
            scale = 1 if variable == "state" else 1 / u.mV
            label = "Measured state" if variable == "state" else f"{variable} [mV]"
            (selected[variable] * scale).plot(ax=ax, x="full_freq_GHz", color=color)
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
