"""Acquire ground- and excited-initial-state T1 traces for every selected qubit.

Run this module directly to acquire both traces and display a single grid figure.
The comparison is intentionally diagnostic: it keeps each acquisition in memory,
shows only one combined figure, and does not update the machine state.
"""

from __future__ import annotations

from importlib import import_module
import logging
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import xarray as xr
from calibration_utils.T1 import Parameters
from calibration_io import CalibrationSaver, current_profile_name
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_config import create_machine
from utils.plotting_settings import FIGURE_SIZE, qubit_grid_locations

from calibrations.core import CalibrationOptions

_t1_module = import_module("calibrations.05_T1")
T1 = _t1_module.T1
DEFAULT_QUBITS = ("q1", "q2", "q3", "q9", "q10")


def _quiet_progress(*_args, **_kwargs) -> None:
    """Keep long all-qubit acquisitions from flooding the terminal."""


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
    grid = QubitGrid(_layout_dataset(layout_qubits), qubit_grid_locations(layout_qubits))
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


def _combined_traces(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Stack the two acquired initial-state traces into one saved dataset."""
    return xr.concat(
        [datasets["g"], datasets["e"]],
        dim=xr.IndexVariable("initial_state", ["g", "e"]),
    )


def save_comparison_result(
    figure,
    traces: xr.Dataset,
    qubit_names: tuple[str, ...],
) -> Path:
    """Save the combined raw traces and the single dashboard figure."""
    saver = CalibrationSaver()
    run_directory = saver.save_xarray(
        "t1_initial_state_comparison",
        traces,
        profile_name=current_profile_name(),
        extra_metadata={
            "qubits": list(qubit_names),
            "initial_states": ["g", "e"],
            "raw_data_saved": True,
            "figure": "figures/t1_initial_state_comparison.png",
        },
    )
    figures_directory = CalibrationSaver().save_figures(
        run_directory,
        {"t1_initial_state_comparison": figure},
    )
    return figures_directory / "t1_initial_state_comparison.png"


def run_initial_state_comparison(
    *,
    simulate: bool = False,
    qubit_names: tuple[str, ...] = DEFAULT_QUBITS,
):
    """Run both preparations one qubit at a time and return a combined grid figure."""
    reference_machine = create_machine()
    missing_qubits = set(qubit_names) - set(reference_machine.qubits)
    if missing_qubits:
        raise ValueError(f"Unknown qubits: {', '.join(sorted(missing_qubits))}")
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
    calibrations: dict[str, dict[str, T1]] = {}
    datasets: dict[str, xr.Dataset] = {}
    for initial_state in ("g", "e"):
        calibrations[initial_state] = {}
        acquired_datasets = []
        for qubit_name in qubit_names:
            parameters = Parameters()
            parameters.qubits = [qubit_name]
            parameters.initial_state = initial_state
            parameters.simulate = simulate
            parameters.use_state_discrimination = True
            parameters.max_wait_time_in_ns = 150e3
            calibration = T1(
                parameters=parameters,
                machine=create_machine(qubit=qubit_name),
                options=options,
                logger=lambda _message: None,
            )
            calibration.run()
            calibrations[initial_state][qubit_name] = calibration
            if not simulate:
                acquired_datasets.append(calibration.results["ds_raw"])
        if not simulate:
            datasets[initial_state] = xr.concat(acquired_datasets, dim="qubit")

    if simulate:
        return calibrations

    figure = plot_initial_state_comparison(
        datasets["g"],
        datasets["e"],
        [reference_machine.qubits[name] for name in qubit_names],
        layout_qubits=compact_layout_qubits(
            [reference_machine.qubits[name] for name in qubit_names],
            reference_machine.qubits.values(),
        ),
    )
    figure_path = save_comparison_result(
        figure,
        _combined_traces(datasets),
        qubit_names,
    )
    print(f"Combined T1 comparison figure saved to {figure_path}")
    plt.show()
    return figure


if __name__ == "__main__":
    run_initial_state_comparison()
