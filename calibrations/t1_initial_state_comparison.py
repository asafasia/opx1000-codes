"""Acquire readout-mitigated ground/excited T1 traces for q1.

The traces are fitted jointly to obtain a shared T1 and equilibrium excited-state
population. The latter is converted to an effective two-level temperature using
q1's configured transition frequency. Results and one diagnostic figure are saved
without updating the machine state.
"""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.T1 import Parameters
from calibration_utils.thermal_relaxation import (
    ThermalRelaxationFit,
    fit_thermal_relaxation,
    mitigated_state_trace,
    thermal_relaxation_model,
)
from calibration_io import CalibrationSaver, current_profile_name
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_config import create_machine
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

from calibrations.core import CalibrationOptions

_t1_module = import_module("calibrations.05_T1")
T1 = _t1_module.T1
DEFAULT_QUBIT = "q1"


def _quiet_progress(*_args, **_kwargs) -> None:
    """Keep the two acquisitions from flooding the terminal."""


_t1_module.progress_counter = _quiet_progress
logging.getLogger("qm").setLevel(logging.WARNING)


def _trace(dataset: xr.Dataset, qubit_name: str) -> tuple[xr.DataArray, str]:
    """Return the directly acquired observable and its display label."""
    selected = dataset.sel(qubit=qubit_name)
    if "state" in selected:
        return selected.state, "State"
    if "I" in selected:
        return selected.I * 1e3, "I [mV]"
    raise ValueError("T1 dataset has neither state-discrimination nor I data.")


def _qubit_names(dataset: xr.Dataset) -> set[str]:
    """Return qubit names present in an acquired dataset."""
    return {str(name) for name in dataset.qubit.values}


def _layout_dataset(qubits: list[Any]) -> xr.Dataset:
    """Create a minimal dataset that lets QubitGrid reserve every layout cell."""
    return xr.Dataset(coords={"qubit": [qubit.name for qubit in qubits]})


def _grid_position(qubit: Any) -> tuple[int, int]:
    """Return the integer device-grid coordinate for a qubit-like object."""
    col, row = str(qubit.grid_location).split(",")
    return int(col), int(row)


def compact_layout_qubits(selected_qubits: Any, all_qubits: Any) -> list[Any]:
    """Keep selected device rows/columns while collapsing unrelated empty rows."""
    selected_positions = [_grid_position(qubit) for qubit in selected_qubits]
    selected_cols = {col for col, _row in selected_positions}
    selected_rows = {row for _col, row in selected_positions}
    col_map = {col: index for index, col in enumerate(sorted(selected_cols))}
    row_map = {row: index for index, row in enumerate(sorted(selected_rows))}

    layout = []
    for qubit in all_qubits:
        col, row = _grid_position(qubit)
        if col not in selected_cols or row not in selected_rows:
            continue
        layout.append(
            SimpleNamespace(
                name=qubit.name,
                grid_location=f"{col_map[col]},{row_map[row]}",
            )
        )
    return layout


def plot_initial_state_comparison(
    ground: xr.Dataset,
    excited: xr.Dataset,
    qubits: Any,
    layout_qubits: Any | None = None,
):
    """Plot acquired traces in the full device qubit grid, leaving empty cells for missing results."""
    qubits = list(qubits)
    layout_qubits = list(layout_qubits or qubits)
    acquired_names = _qubit_names(ground) & _qubit_names(excited)
    grid = QubitGrid(
        _layout_dataset(layout_qubits), qubit_grid_locations(layout_qubits)
    )
    for axis, qubit in grid_iter(grid):
        name = qubit["qubit"]
        if name not in acquired_names:
            axis.set_title(name)
            axis.set_axis_off()
            continue

        ground_trace, label = _trace(ground, name)
        excited_trace, _ = _trace(excited, name)
        ground_trace.plot(ax=axis, marker=".", linestyle="-", label="initial g")
        excited_trace.plot(ax=axis, marker=".", linestyle="-", label="initial e")
        axis.set_title(name)
        axis.set_xlabel("Idle time [ns]")
        axis.set_ylabel(label)
        axis.legend()

    grid.fig.suptitle("T1 initial-state comparison")
    grid.fig.set_size_inches(*FIGURE_SIZE)
    grid.fig.tight_layout()
    return grid.fig


def plot_thermal_relaxation(
    ground: xr.Dataset,
    excited: xr.Dataset,
    *,
    qubit_name: str,
    fit: ThermalRelaxationFit,
):
    """Plot q1's mitigated populations and their joint thermal fit."""
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    ground_trace = mitigated_state_trace(ground, qubit_name)
    excited_trace = mitigated_state_trace(excited, qubit_name)
    ground_trace.plot(ax=axis, marker=".", linestyle="none", label="initial g")
    excited_trace.plot(ax=axis, marker=".", linestyle="none", label="initial e")

    fit_times = np.linspace(
        0.0,
        max(
            float(ground_trace.idle_time.max().item()),
            float(excited_trace.idle_time.max().item()),
        ),
        500,
    )
    for initial_state_code, label in ((0.0, "g fit"), (1.0, "e fit")):
        fitted_population = thermal_relaxation_model(
            np.vstack([fit_times, np.full_like(fit_times, initial_state_code)]),
            fit.t1_ns,
            fit.equilibrium_excited_population,
            fit.initial_ground_trace_population,
            fit.initial_excited_trace_population,
        )
        axis.plot(fit_times, fitted_population, linewidth=2, label=label)

    axis.axhline(
        fit.equilibrium_excited_population,
        color="0.35",
        linestyle="--",
        linewidth=1,
        label=r"$p_e^{eq}$",
    )
    axis.set_title(f"{qubit_name}: thermal T1 from initial g and e")
    axis.set_xlabel("Idle time [ns]")
    axis.set_ylabel("Mitigated excited-state population")
    axis.text(
        0.98,
        0.97,
        (
            rf"$T_1={fit.t1_ns / 1e3:.2f}\pm{fit.t1_error_ns / 1e3:.2f}\,\mu s$"
            "\n"
            rf"$p_e^{{eq}}={fit.equilibrium_excited_population:.4f}"
            rf"\pm{fit.equilibrium_excited_population_error:.4f}$"
            "\n"
            rf"$T_{{eff}}={fit.effective_temperature_kelvin * 1e3:.1f}"
            rf"\pm{fit.effective_temperature_error_kelvin * 1e3:.1f}\,mK$"
        ),
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.legend()
    figure.tight_layout()
    return figure


def _combined_traces(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Stack the two acquired initial-state traces into one saved dataset."""
    return xr.concat(
        [datasets["g"], datasets["e"]],
        dim=xr.IndexVariable("initial_state", ["g", "e"]),
    )


def _add_analysis_to_dataset(
    traces: xr.Dataset,
    fit: ThermalRelaxationFit,
) -> xr.Dataset:
    """Store scalar thermal-fit results beside the acquired traces."""
    return traces.assign(
        thermal_t1_ns=xr.DataArray(
            fit.t1_ns,
            attrs={"long_name": "joint thermal relaxation time", "units": "ns"},
        ),
        thermal_t1_error_ns=xr.DataArray(
            fit.t1_error_ns,
            attrs={"long_name": "joint thermal relaxation time error", "units": "ns"},
        ),
        equilibrium_excited_population=xr.DataArray(
            fit.equilibrium_excited_population,
            attrs={"long_name": "equilibrium excited-state population"},
        ),
        equilibrium_excited_population_error=xr.DataArray(
            fit.equilibrium_excited_population_error,
            attrs={"long_name": "equilibrium excited-state population error"},
        ),
        effective_temperature_kelvin=xr.DataArray(
            fit.effective_temperature_kelvin,
            attrs={"long_name": "effective qubit temperature", "units": "K"},
        ),
        effective_temperature_error_kelvin=xr.DataArray(
            fit.effective_temperature_error_kelvin,
            attrs={"long_name": "effective qubit temperature error", "units": "K"},
        ),
    )


def save_comparison_result(
    figure,
    traces: xr.Dataset,
    *,
    qubit_name: str,
    fit: ThermalRelaxationFit,
) -> Path:
    """Save both population traces, thermal fit results, and the figure."""
    saver = CalibrationSaver()
    run_directory = saver.save_xarray(
        "t1_initial_state_comparison",
        _add_analysis_to_dataset(traces, fit),
        profile_name=current_profile_name(),
        extra_metadata={
            "qubit": qubit_name,
            "initial_states": ["g", "e"],
            "use_state_discrimination": True,
            "use_readout_mitigation": True,
            "thermal_fit": asdict(fit),
            "figure": "figures/t1_initial_state_comparison.png",
        },
    )
    figures_directory = saver.save_figures(
        run_directory,
        {"t1_initial_state_comparison": figure},
    )
    return figures_directory / "t1_initial_state_comparison.png"


def run_initial_state_comparison(
    *,
    simulate: bool = False,
    qubit_name: str = DEFAULT_QUBIT,
):
    """Acquire q1 from both initial states and report its effective temperature."""
    if qubit_name != DEFAULT_QUBIT:
        raise ValueError(f"This experiment is restricted to {DEFAULT_QUBIT}.")

    reference_machine = create_machine(qubit=qubit_name)
    qubit = reference_machine.qubits[qubit_name]
    options = CalibrationOptions(
        save_raw_data=False,
        save_analysis_result=False,
        save_figures=False,
        analyse_data=False,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )
    calibrations: dict[str, T1] = {}
    datasets: dict[str, xr.Dataset] = {}
    for initial_state in ("g", "e"):
        parameters = Parameters()
        parameters.qubits = [qubit_name]
        parameters.initial_state = initial_state
        parameters.reset_type = "active"

        parameters.simulate = simulate
        parameters.use_state_discrimination = True
        parameters.use_readout_mitigation = True
        parameters.max_wait_time_in_ns = 250e3
        parameters.num_shots = 1000
        parameters.wait_time_num_points = 300

        calibration = T1(
            parameters=parameters,
            machine=create_machine(qubit=qubit_name),
            options=options,
            logger=lambda _message: None,
        )
        calibration.run()
        calibrations[initial_state] = calibration
        if not simulate:
            datasets[initial_state] = calibration.results["ds_raw"]

    if simulate:
        return calibrations

    fit = fit_thermal_relaxation(
        datasets["g"],
        datasets["e"],
        qubit_name=qubit_name,
        qubit_frequency_hz=float(qubit.f_01),
    )
    figure = plot_thermal_relaxation(
        datasets["g"],
        datasets["e"],
        qubit_name=qubit_name,
        fit=fit,
    )
    figure_path = save_comparison_result(
        figure,
        _combined_traces(datasets),
        qubit_name=qubit_name,
        fit=fit,
    )
    print(
        f"{qubit_name}: T1 = {fit.t1_ns / 1e3:.3f} +/- "
        f"{fit.t1_error_ns / 1e3:.3f} us, "
        f"p_e(eq) = {fit.equilibrium_excited_population:.5f} +/- "
        f"{fit.equilibrium_excited_population_error:.5f}, "
        f"T_eff = {fit.effective_temperature_kelvin * 1e3:.2f} +/- "
        f"{fit.effective_temperature_error_kelvin * 1e3:.2f} mK"
    )
    print(f"Thermal T1 figure saved to {figure_path}")
    plt.show()
    return fit, figure


if __name__ == "__main__":
    run_initial_state_comparison()
