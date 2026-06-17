"""Sweep gate length by running the full DRAG workflow at each length."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from calibration_io import CalibrationSaver
from calibrations_v2 import CalibrationOptions
from profiles import Profile, validate_profile
from profiles.loader import _select_qubit
from workflows.drag_workflow import DragWorkflow, DragWorkflowParameters


def align_gate_lengths_to_4ns(lengths_ns) -> np.ndarray:
    """Round requested pulse lengths up to the next QOP-valid 4 ns grid point."""
    lengths = np.asarray(lengths_ns, dtype=int)
    if np.any(lengths <= 0):
        raise ValueError("Gate lengths must be positive integers.")
    aligned = lengths + (-lengths % 4)
    return np.unique(aligned)


def gate_lengths_from_range(
    start_ns: int,
    stop_ns: int,
    step_ns: int,
    *,
    align_to_4ns: bool = True,
) -> np.ndarray:
    """Create a gate-length scan, optionally snapping to QOP-valid 4 ns lengths."""
    lengths = np.arange(start_ns, stop_ns + 1, step_ns, dtype=int)
    if align_to_4ns:
        return align_gate_lengths_to_4ns(lengths)
    return lengths


@dataclass
class GateLengthDragWorkflowSweepParameters:
    """Parameters for sweeping gate length through the full DRAG workflow."""

    qubit: str = "q9"
    profile_name: str = "single_qubit"
    profile_documents: dict[str, Any] | None = None
    gate_lengths_ns: np.ndarray = field(
        default_factory=lambda: gate_lengths_from_range(6, 100, 10)
    )
    save_results: bool = True
    plot_results: bool = True
    workflow: DragWorkflowParameters = field(default_factory=DragWorkflowParameters)


class GateLengthDragWorkflowSweep:
    """Run Power Rabi, DRAG calibration, and DRAG RB for each gate length."""

    name = "gate_length_drag_workflow_sweep"

    def __init__(
        self,
        parameters: GateLengthDragWorkflowSweepParameters | None = None,
        *,
        saver: CalibrationSaver | None = None,
        workflow_class: type[DragWorkflow] = DragWorkflow,
    ) -> None:
        self.parameters = parameters or GateLengthDragWorkflowSweepParameters()
        self.saver = saver or CalibrationSaver()
        self.workflow_class = workflow_class
        self.profile_documents = self._load_profile_documents()
        self.results: dict[str, Any] = {}
        self.workflows: list[DragWorkflow] = []
        self.run_directory: Path | None = None

    def _load_profile_documents(self) -> dict[str, Any]:
        if self.parameters.profile_documents is not None:
            documents = deepcopy(self.parameters.profile_documents)
            if documents["manifest"].get("build_mode") == "single_qubit":
                documents = _select_qubit(documents, self.parameters.qubit)
            validate_profile(documents)
            return documents
        return Profile(
            self.parameters.profile_name,
            qubit=self.parameters.qubit,
        ).load()

    def run(self) -> dict[str, Any]:
        lengths = np.asarray(self.parameters.gate_lengths_ns, dtype=int)
        if np.any(lengths <= 0):
            raise ValueError("gate_lengths_ns must contain only positive integers.")
        invalid_lengths = lengths[lengths % 4 != 0]
        if invalid_lengths.size:
            raise ValueError(
                "QOP pulse lengths must be multiples of 4 ns. Invalid lengths: "
                f"{invalid_lengths.tolist()}. Use gate_lengths_from_range(..., "
                "align_to_4ns=True) or align_gate_lengths_to_4ns(...)."
            )
        rows: list[dict[str, Any]] = []
        self.workflows = []

        try:
            for index, length_ns in enumerate(lengths, start=1):
                print(
                    f"\n=== Gate-length DRAG workflow {index}/{len(lengths)}: "
                    f"length={length_ns} ns ==="
                )
                workflow_parameters = deepcopy(self.parameters.workflow)
                workflow_parameters.qubit = self.parameters.qubit
                workflow_parameters.profile_name = self.parameters.profile_name
                workflow_parameters.profile_documents = deepcopy(self.profile_documents)
                workflow_parameters.gate_length_ns = int(length_ns)
                workflow_parameters.rabi.operation = "x180_drag"
                workflow_parameters.rb.gate_family = "drag"

                workflow = self.workflow_class(parameters=workflow_parameters)
                workflow.run()
                self.workflows.append(workflow)
                rows.extend(self._rows_from_workflow(int(length_ns), workflow))
        except KeyboardInterrupt:
            self._finalize_results(lengths, rows, interrupted=True)
            if rows and self.run_directory is not None:
                print(
                    "Interrupted gate-length DRAG workflow sweep; partial results "
                    f"saved to {self.run_directory}"
                )
            raise

        self._finalize_results(lengths, rows, interrupted=False)
        return self.results

    def _finalize_results(
        self,
        lengths: np.ndarray,
        rows: list[dict[str, Any]],
        *,
        interrupted: bool,
    ) -> None:
        if not rows:
            self.results = {
                "gate_length_ns": lengths,
                "rows": [],
                "interrupted": interrupted,
                "completed_points": 0,
                "planned_points": int(len(lengths)),
            }
            return
        self.results = self._aggregate_results(lengths, rows)
        self.results["interrupted"] = interrupted
        self.results["completed_points"] = len({row["gate_length_ns"] for row in rows})
        self.results["planned_points"] = int(len(lengths))
        if self.parameters.plot_results:
            self.results["figures"] = self._plot_results()
        if self.parameters.save_results:
            self.run_directory = self.save_results()

    def _rows_from_workflow(
        self, length_ns: int, workflow: DragWorkflow
    ) -> list[dict[str, Any]]:
        rb = workflow.calibrations.get("rb")
        if rb is None:
            return [self._failed_row(length_ns)]

        rows = []
        for qubit_name, fit in rb.results.get("fit_results", {}).items():
            error_per_gate = float(fit["error_per_gate"])
            fidelity_std = float(fit.get("fidelity_std", np.nan))
            rows.append(
                {
                    "gate_length_ns": int(length_ns),
                    "qubit": qubit_name,
                    "success": bool(fit.get("success", False)),
                    "error_per_gate": error_per_gate,
                    "error_per_gate_std": float(
                        fit.get("error_per_gate_std", fidelity_std)
                    ),
                    "fidelity": float(fit.get("fidelity", 1.0 - error_per_gate)),
                    "fidelity_std": fidelity_std,
                }
            )
        return rows or [self._failed_row(length_ns)]

    def _failed_row(self, length_ns: int) -> dict[str, Any]:
        return {
            "gate_length_ns": int(length_ns),
            "qubit": self.parameters.qubit,
            "success": False,
            "error_per_gate": np.nan,
            "error_per_gate_std": np.nan,
            "fidelity": np.nan,
            "fidelity_std": np.nan,
        }

    def _aggregate_results(
        self,
        lengths: np.ndarray,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        qubits = sorted({row["qubit"] for row in rows})
        fidelity = np.full((len(qubits), len(lengths)), np.nan)
        fidelity_std = np.full_like(fidelity, np.nan)
        error_per_gate = np.full_like(fidelity, np.nan)
        error_per_gate_std = np.full_like(fidelity, np.nan)
        success = np.zeros((len(qubits), len(lengths)), dtype=bool)
        qubit_index = {qubit: index for index, qubit in enumerate(qubits)}
        length_index = {int(length): index for index, length in enumerate(lengths)}

        for row in rows:
            qi = qubit_index[row["qubit"]]
            li = length_index[row["gate_length_ns"]]
            fidelity[qi, li] = row["fidelity"]
            fidelity_std[qi, li] = row["fidelity_std"]
            error_per_gate[qi, li] = row["error_per_gate"]
            error_per_gate_std[qi, li] = row["error_per_gate_std"]
            success[qi, li] = row["success"]

        best_gate_length_ns = np.full(len(qubits), np.nan)
        best_fidelity = np.full(len(qubits), np.nan)
        for qubit, qi in qubit_index.items():
            if np.all(np.isnan(fidelity[qi])):
                continue
            best_index = int(np.nanargmax(fidelity[qi]))
            best_gate_length_ns[qi] = lengths[best_index]
            best_fidelity[qi] = fidelity[qi, best_index]

        return {
            "gate_length_ns": lengths,
            "qubit": np.asarray(qubits, dtype=str),
            "fidelity": fidelity,
            "fidelity_std": fidelity_std,
            "error_per_gate": error_per_gate,
            "error_per_gate_std": error_per_gate_std,
            "success": success,
            "best_gate_length_ns": best_gate_length_ns,
            "best_fidelity": best_fidelity,
            "rows": rows,
        }

    def _plot_results(self) -> dict[str, Any]:
        lengths = self.results["gate_length_ns"]
        figures = {}
        for index, qubit in enumerate(self.results["qubit"]):
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.errorbar(
                lengths,
                self.results["fidelity"][index],
                yerr=self.results["fidelity_std"][index],
                marker="o",
                capsize=3,
                label="DRAG workflow RB fidelity",
            )
            if not np.isnan(self.results["best_gate_length_ns"][index]):
                axis.axvline(
                    self.results["best_gate_length_ns"][index],
                    color="tab:red",
                    linestyle="--",
                    label=f"best length = {self.results['best_gate_length_ns'][index]:.0f} ns",
                )
            axis.set_title(f"{qubit} gate-length DRAG workflow sweep")
            axis.set_xlabel("Gate length [ns]")
            axis.set_ylabel("Single-qubit RB fidelity")
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figures[f"{qubit}_gate_length_drag_workflow_sweep"] = figure
        plt.show()
        return figures

    def save_results(self) -> Path:
        run_directory = self.saver.save(
            self.name,
            sweep={
                "gate_length_ns": self.results["gate_length_ns"],
                "qubit": self.results["qubit"],
            },
            results={
                "fidelity": self.results["fidelity"],
                "fidelity_std": self.results["fidelity_std"],
                "error_per_gate": self.results["error_per_gate"],
                "error_per_gate_std": self.results["error_per_gate_std"],
                "success": self.results["success"],
                "best_gate_length_ns": self.results["best_gate_length_ns"],
                "best_fidelity": self.results["best_fidelity"],
                "interrupted": np.asarray(self.results.get("interrupted", False)),
                "completed_points": np.asarray(self.results.get("completed_points", 0)),
                "planned_points": np.asarray(self.results.get("planned_points", 0)),
            },
            profile_name=self.parameters.profile_name,
        )
        summary = {
            "interrupted": bool(self.results.get("interrupted", False)),
            "completed_points": int(self.results.get("completed_points", 0)),
            "planned_points": int(self.results.get("planned_points", 0)),
            "qubit": self.results["qubit"].tolist(),
            "gate_length_ns": self.results["gate_length_ns"].tolist(),
            "best_gate_length_ns": self.results["best_gate_length_ns"].tolist(),
            "best_fidelity": self.results["best_fidelity"].tolist(),
            "fidelity_std": self.results["fidelity_std"].tolist(),
            "rows": self.results["rows"],
        }
        with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
        if self.results.get("figures"):
            self.saver.save_figures(run_directory, self.results["figures"])
        print(f"Gate-length DRAG workflow sweep saved to {run_directory}")
        return run_directory


def default_parameters() -> GateLengthDragWorkflowSweepParameters:
    """Return a 6..100 ns by 10 ns requested sweep snapped to valid 4 ns lengths."""
    parameters = GateLengthDragWorkflowSweepParameters()
    parameters.gate_lengths_ns = gate_lengths_from_range(6, 100, 10)
    parameters.workflow = DragWorkflowParameters()
    parameters.workflow.options = CalibrationOptions(
        save_raw_data=False,
        save_figures=False,
        analyse_data=True,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )
    parameters.workflow.connect_before_run = True
    parameters.workflow.close_existing_qms = True
    parameters.workflow.rabi.use_state_discrimination = True
    parameters.workflow.rabi.reset_type = "active"
    parameters.workflow.rabi.operation = "x180_drag"
    parameters.workflow.rabi.num_shots = 500
    parameters.workflow.rabi.transition = "ge"
    parameters.workflow.rabi.pi_repetitions = 4
    parameters.workflow.drag.use_state_discrimination = True
    parameters.workflow.drag.reset_type = "active"
    parameters.workflow.rb.use_state_discrimination = True
    parameters.workflow.rb.reset_type = "active"
    parameters.workflow.rb.gate_family = "drag"
    parameters.workflow.rb.max_circuit_depth = 1024
    parameters.workflow.rb.delta_clifford = 2
    parameters.workflow.rb.num_random_sequences = 30
    parameters.workflow.rb.num_shots = 100
    parameters.workflow.rb.log_scale = True
    parameters.workflow.rb.use_strict_timing = True
    return parameters


if __name__ == "__main__":
    parameters = default_parameters()
    parameters.qubit = "q9"
    parameters.profile_name = "single_qubit"
    parameters.gate_lengths_ns = gate_lengths_from_range(6, 100, 10)
    parameters.save_results = True
    parameters.plot_results = True

    sweep = GateLengthDragWorkflowSweep(parameters)
    sweep.run()
