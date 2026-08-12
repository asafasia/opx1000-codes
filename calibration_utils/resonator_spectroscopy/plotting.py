from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from utils.plotting_settings import (
    FIGURE_SIZE,
    CalibrationPlot,
    add_calibration_parameter_box,
    format_readout_parameter_lines,
    qubit_grid_locations,
)
from utils.rabi_amplitude import qubit_amplitude_to_rabi_frequency_hz

u = unit(coerce_to_integer=True)


def _frequency_to_hz(frequency: float) -> float:
    """Accept GHz-style inputs such as 6.875 or explicit Hz inputs."""
    frequency = float(frequency)
    if abs(frequency) < 100:
        return frequency * u.GHz
    return frequency


def _full_frequency_axis(selected: xr.Dataset, qubit: AnyTransmon) -> np.ndarray:
    if "full_freq" in selected.coords:
        return np.asarray(selected.full_freq.values, dtype=float)
    return np.asarray(selected.detuning.values, dtype=float) + float(qubit.resonator.RF_frequency)


def _nearest_frequency_index(selected: xr.Dataset, qubit: AnyTransmon, frequency_hz: float) -> int:
    frequencies = _full_frequency_axis(selected, qubit)
    if frequencies.size == 0:
        raise ValueError("Cannot select a frequency from an empty detuning sweep.")
    return int(np.nanargmin(np.abs(frequencies - frequency_hz)))


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


def _operation_name(qubit_operation: str) -> str:
    return "x180" if qubit_operation == "x180_const" else qubit_operation


def _state_response_variables() -> dict[str, tuple[str, str, str, str, str]]:
    return {
        "g": ("Ig", "Qg", "ground_IQ_abs", "ground_phase", "Ground"),
        "e": ("Im", "Qm", "mixed_IQ_abs", "mixed_phase", "Driven"),
        "f": ("If", "Qf", "f_IQ_abs", "f_phase", "F"),
    }


def _available_states(ds: xr.Dataset) -> list[str]:
    return [
        state
        for state, (i_name, q_name, *_rest) in _state_response_variables().items()
        if {i_name, q_name}.issubset(ds.data_vars)
    ]


def _available_processed_states(ds: xr.Dataset) -> list[str]:
    return [
        state
        for state, (_i_name, _q_name, iq_abs_name, _phase_name, _label) in _state_response_variables().items()
        if iq_abs_name in ds.data_vars
    ]


def _format_qubit_drive_line(
    qubit,
    qubit_operation: str,
    saturation_amplitude_factor: float,
    saturation_lead_time_in_ns: Optional[int],
) -> str:
    operation = _operation_name(qubit_operation)
    operations = getattr(getattr(qubit, "xy", None), "operations", {})
    pulse = operations.get(operation) if hasattr(operations, "get") else None
    parts = [
        f"{qubit.name}: driven operation={qubit_operation}",
        f"amplitude factor={saturation_amplitude_factor:g}",
    ]
    if saturation_lead_time_in_ns is not None and qubit_operation == "saturation":
        parts.append(f"lead time={float(saturation_lead_time_in_ns):g} ns")
    if pulse is not None:
        length = getattr(pulse, "length", None)
        configured_amplitude = getattr(pulse, "amplitude", None)
        if length is not None:
            parts.append(f"drive length={float(length):g} ns")
        if configured_amplitude is not None:
            played_amplitude = float(configured_amplitude) * saturation_amplitude_factor
            rabi_frequency_part = ""
            try:
                rabi_frequency_mhz = (
                    qubit_amplitude_to_rabi_frequency_hz(played_amplitude, qubit) / u.MHz
                )
                rabi_frequency_part = f", {float(rabi_frequency_mhz):.3f} MHz"
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            parts.append(
                f"drive amp={1e3 * played_amplitude:.3f} mV{rabi_frequency_part} "
                f"(configured {1e3 * float(configured_amplitude):.3f} mV)"
            )
    return " | ".join(parts)


def _resonator_parameter_lines(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    qubit_operation: str,
    saturation_amplitude_factor: float,
    saturation_lead_time_in_ns: Optional[int],
) -> list[str]:
    lines = ["Parameters"]
    if "detuning" in ds.coords:
        detuning = np.asarray(ds.detuning.values, dtype=float)
        if detuning.size:
            span_mhz = (float(np.nanmax(detuning)) - float(np.nanmin(detuning))) / u.MHz
            if detuning.size > 1:
                step_mhz = abs(float(detuning[1] - detuning[0])) / u.MHz
                lines.append(f"frequency span={span_mhz:g} MHz, step={step_mhz:g} MHz")
            else:
                lines.append(f"frequency span={span_mhz:g} MHz")
    if "n_runs" in ds.sizes:
        lines.append(f"num shots={ds.sizes['n_runs']}")
    lines.extend(format_readout_parameter_lines(qubits))
    lines.extend(
        _format_qubit_drive_line(
            qubit,
            qubit_operation,
            saturation_amplitude_factor,
            saturation_lead_time_in_ns,
        )
        for qubit in qubits
    )
    return lines


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax1, qubit in grid_iter(grid):
        selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit]
        for state in _available_processed_states(selected):
            _i_name, _q_name, _iq_abs_name, phase_name, label = _state_response_variables()[state]
            selected[phase_name].plot(ax=ax1, x="full_freq_GHz", label=label)
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        ax1.legend()
    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state phase")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    qubit_operation: str = "saturation",
    saturation_amplitude_factor: float = 1.0,
    saturation_lead_time_in_ns: Optional[int] = None,
):
    """
    Plot mean resonator responses, shot-cloud separation, and readout fidelity.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
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
    height_ratios = [3, 1, 1] * rows
    fig, axes = plt.subplots(
        3 * rows,
        columns,
        figsize=FIGURE_SIZE,
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios},
    )

    used_axes = set()
    for qubit, (row, column) in zip(qubits, locations):
        spectrum_ax = axes[3 * row, column]
        difference_ax = axes[3 * row + 1, column]
        fidelity_ax = axes[3 * row + 2, column]
        used_axes.update(
            {
                (3 * row, column),
                (3 * row + 1, column),
                (3 * row + 2, column),
            }
        )

        selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).sel(qubit=qubit.name)
        separation = selected.IQ_separation
        max_separation_index = int(separation.argmax(dim="detuning").values)
        max_separation_frequency_ghz = float(
            selected.full_freq_GHz.isel(detuning=max_separation_index).values
        )
        current_frequency_ghz = float(qubit.resonator.RF_frequency) / u.GHz
        max_separation_label = (
            f"New resonance (maximum normalized IQ separation): {max_separation_frequency_ghz:.6f} GHz"
        )
        current_frequency_label = f"Current resonance: {current_frequency_ghz:.6f} GHz"

        for state in _available_processed_states(selected):
            _i_name, _q_name, iq_abs_name, _phase_name, label = _state_response_variables()[state]
            (selected[iq_abs_name] / u.mV).plot(
                ax=spectrum_ax, x="full_freq_GHz", label=label
            )
        spectrum_ax.axvline(
            current_frequency_ghz,
            color="black",
            linestyle=":",
            label=current_frequency_label,
        )
        spectrum_ax.axvline(
            max_separation_frequency_ghz,
            color="tab:red",
            linestyle="--",
            label=max_separation_label,
        )
        spectrum_ax.set_title(qubit.name)
        spectrum_ax.set_xlabel("")
        spectrum_ax.set_ylabel(r"$|IQ|$ [mV]")
        spectrum_ax.legend()

        separation.plot(
            ax=difference_ax,
            x="full_freq_GHz",
            color="tab:blue",
            label="Normalized IQ separation",
        )
        if "readout_fidelity" in selected.data_vars:
            selected.readout_fidelity.plot(
                ax=fidelity_ax,
                x="full_freq_GHz",
                color="tab:orange",
                label="Readout fidelity",
            )
        else:
            fidelity_ax.text(
                0.5,
                0.5,
                "Fidelity unavailable",
                transform=fidelity_ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
        state_pairs = (
            set(str(value) for value in selected.pairwise_IQ_separation.state_pair.values)
            if "pairwise_IQ_separation" in selected.data_vars
            else set()
        )
        if len(state_pairs) > 1:
            for state_pair in selected.pairwise_IQ_separation.state_pair.values:
                selected.pairwise_IQ_separation.sel(state_pair=state_pair).plot(
                    ax=difference_ax,
                    x="full_freq_GHz",
                    linestyle=":",
                    alpha=0.6,
                    label=f"{state_pair} separation",
                )
        difference_ax.axvline(
            current_frequency_ghz,
            color="black",
            linestyle=":",
            label=current_frequency_label,
        )
        difference_ax.axvline(
            max_separation_frequency_ghz,
            color="tab:red",
            linestyle="--",
            label=max_separation_label,
        )
        fidelity_ax.axvline(
            current_frequency_ghz,
            color="black",
            linestyle=":",
            label=current_frequency_label,
        )
        fidelity_ax.axvline(
            max_separation_frequency_ghz,
            color="tab:red",
            linestyle="--",
            label=max_separation_label,
        )
        difference_ax.set_xlabel("")
        difference_ax.set_ylabel("IQ separation / pooled std")
        difference_ax.legend()
        fidelity_ax.set_xlabel("RF frequency [GHz]")
        fidelity_ax.set_ylabel("Optimal discrimination fidelity [%]")
        fidelity_ax.set_ylim(75, 100)
        fidelity_ax.legend()
        _add_detuning_axis(spectrum_ax, current_frequency_ghz)
        _add_detuning_axis(difference_ax, current_frequency_ghz)
        _add_detuning_axis(fidelity_ax, current_frequency_ghz)

    for row in range(3 * rows):
        for column in range(columns):
            if (row, column) not in used_axes:
                axes[row, column].set_visible(False)

    fig.suptitle("Resonator spectroscopy")
    parameter_lines = _resonator_parameter_lines(
        ds,
        qubits,
        qubit_operation,
        saturation_amplitude_factor,
        saturation_lead_time_in_ns,
    )
    add_calibration_parameter_box(fig, parameter_lines, gid="resonator_spectroscopy_parameters")
    calibration_plot = CalibrationPlot(fig)
    calibration_plot.add_timestamp()
    calibration_plot.tight_layout_for_parameters(len(parameter_lines))
    return fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    if fit:
        fitted_data = lorentzian_dip(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    selected = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit]
    for state in _available_processed_states(selected):
        _i_name, _q_name, iq_abs_name, _phase_name, label = _state_response_variables()[state]
        (selected[iq_abs_name] / u.mV).plot(ax=ax, x="full_freq_GHz", label=label)
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    if fitted_data is not None:
        ax.plot(ds.full_freq.loc[qubit] / u.GHz, fitted_data / u.mV, "k--", label="Fit")
    ax.legend()


def plot_iq_response(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot ground and mixed-state resonator trajectories in the IQ plane."""
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    for ax, qubit in grid_iter(grid):
        selected = ds.loc[qubit]
        for state in _available_states(selected):
            i_name, q_name, _iq_abs_name, _phase_name, label = _state_response_variables()[state]
            ax.plot(
                selected[i_name] / u.mV,
                selected[q_name] / u.mV,
                ".-",
                label=label,
                markersize=3,
            )
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.axis("equal")
        ax.legend()

    grid.fig.suptitle("Resonator spectroscopy: ground and mixed-state IQ response")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_iq_blobs_for_frequency(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    frequency: float,
) -> Figure:
    """Plot shot-level resonator IQ clouds at the nearest point to ``frequency``.

    ``frequency`` may be passed in GHz, for example ``6.875``, or in Hz, for
    example ``6_875_000_000``.
    """
    frequency_hz = _frequency_to_hz(frequency)
    grid = QubitGrid(ds, qubit_grid_locations(qubits))
    selected_summaries = []

    for ax, qubit_ref in grid_iter(grid):
        qubit_name = qubit_ref["qubit"]
        qubit = next(q for q in qubits if q.name == qubit_name)
        selected = ds.sel(qubit=qubit_name)
        frequency_index = _nearest_frequency_index(selected, qubit, frequency_hz)
        point = selected.isel(detuning=frequency_index)
        frequencies = _full_frequency_axis(selected, qubit)
        selected_frequency_hz = float(frequencies[frequency_index])
        selected_detuning_hz = float(point.detuning.values)
        selected_summaries.append(
            f"{qubit_name}: index={frequency_index}, "
            f"freq={selected_frequency_hz / u.GHz:.6f} GHz, "
            f"detuning={selected_detuning_hz / u.MHz:.3f} MHz"
        )

        colors = {"g": "tab:blue", "e": "tab:orange", "f": "tab:green"}
        for state in _available_states(point):
            i_name, q_name, _iq_abs_name, _phase_name, label = _state_response_variables()[state]
            ax.plot(
                point[i_name] / u.mV,
                point[q_name] / u.mV,
                ".",
                alpha=0.4,
                markersize=3,
                label=label,
            )
            ax.plot(
                float(point[i_name].mean(dim="n_runs")) / u.mV,
                float(point[q_name].mean(dim="n_runs")) / u.mV,
                "o",
                color=colors[state],
                markeredgecolor="black",
                label=f"{label} center",
            )
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.axis("equal")
        ax.set_title(
            f"{qubit_name}: nearest to {frequency_hz / u.GHz:.6f} GHz\n"
            f"index {frequency_index}, {selected_frequency_hz / u.GHz:.6f} GHz"
        )
        ax.legend(fontsize="small")

    grid.fig.suptitle("Resonator spectroscopy IQ blobs at selected frequency")
    grid.fig.set_size_inches(1.15 * FIGURE_SIZE[0], 1.15 * FIGURE_SIZE[1])
    add_calibration_parameter_box(
        grid.fig,
        ["Selected frequency", *selected_summaries],
        gid="resonator_spectroscopy_frequency_iq_parameters",
    )
    calibration_plot = CalibrationPlot(grid.fig)
    calibration_plot.add_timestamp()
    calibration_plot.tight_layout_for_parameters(len(selected_summaries) + 1)
    return grid.fig
