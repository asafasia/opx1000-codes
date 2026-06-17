"""Sweep DRAG beta by running randomized benchmarking at each value."""

from __future__ import annotations

import importlib
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from calibration_io import CalibrationSaver
from calibration_utils.single_qubit_randomized_benchmarking import (
    Parameters as RandomizedBenchmarkingParameters,
)
from calibrations_v2 import CalibrationOptions
from profiles import Profile, validate_profile
from profiles.loader import _select_qubit
from quam_config import Quam
from quam_config.create_machine_from_profile import create_machine_from_profile
from workflows.drag_workflow import InMemoryProfile, _set_profile_path

SingleQubitRandomizedBenchmarking = importlib.import_module(
    "calibrations_v2.11a_single_qubit_randomized_benchmarking"
).SingleQubitRandomizedBenchmarking


@dataclass
class DragSweepParameters:
    """Parameters for sweeping DRAG beta and validating each point with RB."""

    qubit: str = "q9"
    profile_name: str = "single_qubit"
    profile_documents: dict[str, Any] | None = None
    drag_operation: str = "x180_drag"
    beta_values: np.ndarray = field(default_factory=lambda: np.linspace(-1.0, 2.0, 13))
    connect_before_run: bool = True
    close_existing_qms: bool = True
    save_results: bool = True
    plot_results: bool = True
    rb: RandomizedBenchmarkingParameters = field(
        default_factory=RandomizedBenchmarkingParameters
    )
    rb_options: CalibrationOptions = field(
        default_factory=lambda: CalibrationOptions(
            save_raw_data=False,
            save_figures=False,
            analyse_data=True,
            plot_data=False,
            update_state=False,
            propose_profile_update=False,
            apply_profile_update=False,
        )
    )


class DragSweep:
    """Run RB while scanning the beta of a DRAG pulse."""

    name = "drag_sweep"

    def __init__(
        self,
        parameters: DragSweepParameters | None = None,
        *,
        saver: CalibrationSaver | None = None,
    ) -> None:
        self.parameters = parameters or DragSweepParameters()
        self.saver = saver or CalibrationSaver()
        self.profile_documents = self._load_profile_documents()
        self.results: dict[str, Any] = {}
        self.run_directory: Path | None = None
        self.machine: Quam | None = None

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

    def _beta_update_path(self, qubit_name: str) -> str:
        qubit_profile = self.profile_documents["qubits"]["qubits"][qubit_name]
        pulse_name = qubit_profile["operations"].get(self.parameters.drag_operation)
        pulse_profile = (
            self.profile_documents["pulses"]["pulses"][qubit_name].get(pulse_name)
            if pulse_name
            else None
        )
        if pulse_profile is None or pulse_profile.get("type") != "drag":
            raise ValueError(
                f"{qubit_name} operation {self.parameters.drag_operation!r} does not "
                "map to a DRAG pulse in the selected profile."
            )
        return f"pulses.json.pulses.{qubit_name}.{pulse_name}.beta"

    def _documents_for_beta(self, beta: float) -> dict[str, Any]:
        documents = deepcopy(self.profile_documents)
        _set_profile_path(
            documents,
            self._beta_update_path(self.parameters.qubit),
            float(beta),
        )
        validate_profile(documents)
        return documents

    def _build_machine(self, documents: Mapping[str, Any]) -> Quam:
        profile = InMemoryProfile(
            documents,
            name=self.parameters.profile_name,
            qubit=self.parameters.qubit,
        )
        machine = create_machine_from_profile(profile, save=False)
        if self.parameters.connect_before_run:
            machine.connect()
            if self.parameters.close_existing_qms:
                machine.qmm.close_all_qms()
        self.machine = machine
        return machine

    def run(self) -> dict[str, Any]:
        """Run the full beta sweep and optionally save/plot the summary."""
        beta_values = np.asarray(self.parameters.beta_values, dtype=float)
        rb_parameters = self.parameters.rb
        rb_parameters.gate_family = "drag"

        rows: list[dict[str, Any]] = []
        calibrations = []
        try:
            for index, beta in enumerate(beta_values, start=1):
                print(f"\n=== DRAG sweep {index}/{len(beta_values)}: beta={beta:.6g} ===")
                documents = self._documents_for_beta(float(beta))
                machine = self._build_machine(documents)
                calibration = SingleQubitRandomizedBenchmarking(
                    parameters=rb_parameters,
                    machine=machine,
                    options=self.parameters.rb_options,
                )
                status = calibration.run()
                calibrations.append(calibration)
                rows.extend(self._rows_from_calibration(beta, calibration, status))
        except KeyboardInterrupt:
            self._finalize_results(beta_values, rows, calibrations, interrupted=True)
            if rows and self.run_directory is not None:
                print(f"Interrupted DRAG sweep; partial results saved to {self.run_directory}")
            raise

        self._finalize_results(beta_values, rows, calibrations, interrupted=False)
        return self.results

    def _finalize_results(
        self,
        beta_values: np.ndarray,
        rows: list[dict[str, Any]],
        calibrations: list[Any],
        *,
        interrupted: bool,
    ) -> None:
        if not rows:
            self.results = {
                "beta": beta_values,
                "rows": [],
                "interrupted": interrupted,
                "completed_points": 0,
                "planned_points": int(len(beta_values)),
                "calibrations": calibrations,
            }
            return
        self.results = self._aggregate_results(beta_values, rows)
        self.results["interrupted"] = interrupted
        self.results["completed_points"] = len({row["beta"] for row in rows})
        self.results["planned_points"] = int(len(beta_values))
        if self.parameters.plot_results:
            self.results["figures"] = self._plot_results()
        if self.parameters.save_results:
            self.run_directory = self.save_results()
        self.results["calibrations"] = calibrations

    def _rows_from_calibration(
        self, beta: float, calibration: Any, status: Any
    ) -> list[dict[str, Any]]:
        rows = []
        fit_results = calibration.results.get("fit_results", {})
        for qubit_name, fit in fit_results.items():
            error_per_gate = float(fit["error_per_gate"])
            fidelity_std = float(fit.get("fidelity_std", np.nan))
            rows.append(
                {
                    "beta": float(beta),
                    "qubit": qubit_name,
                    "success": bool(fit.get("success", False)),
                    "error_per_gate": error_per_gate,
                    "error_per_gate_std": float(
                        fit.get("error_per_gate_std", fidelity_std)
                    ),
                    "fidelity": float(1.0 - error_per_gate),
                    "fidelity_std": fidelity_std,
                    "status_mode": status.mode,
                }
            )
        if not rows:
            rows.append(
                {
                    "beta": float(beta),
                    "qubit": self.parameters.qubit,
                    "success": False,
                    "error_per_gate": np.nan,
                    "error_per_gate_std": np.nan,
                    "fidelity": np.nan,
                    "fidelity_std": np.nan,
                    "status_mode": status.mode,
                }
            )
        return rows

    def _aggregate_results(
        self,
        beta_values: np.ndarray,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        qubits = sorted({row["qubit"] for row in rows})
        fidelity = np.full((len(qubits), len(beta_values)), np.nan)
        fidelity_std = np.full_like(fidelity, np.nan)
        error_per_gate = np.full_like(fidelity, np.nan)
        error_per_gate_std = np.full_like(fidelity, np.nan)
        success = np.zeros((len(qubits), len(beta_values)), dtype=bool)
        qubit_index = {qubit: index for index, qubit in enumerate(qubits)}
        beta_index = {float(beta): index for index, beta in enumerate(beta_values)}

        for row in rows:
            qi = qubit_index[row["qubit"]]
            bi = beta_index[row["beta"]]
            fidelity[qi, bi] = row["fidelity"]
            fidelity_std[qi, bi] = row["fidelity_std"]
            error_per_gate[qi, bi] = row["error_per_gate"]
            error_per_gate_std[qi, bi] = row["error_per_gate_std"]
            success[qi, bi] = row["success"]

        best_beta = np.full(len(qubits), np.nan)
        best_fidelity = np.full(len(qubits), np.nan)
        for qubit, qi in qubit_index.items():
            if np.all(np.isnan(fidelity[qi])):
                continue
            best_index = int(np.nanargmax(fidelity[qi]))
            best_beta[qi] = beta_values[best_index]
            best_fidelity[qi] = fidelity[qi, best_index]

        return {
            "beta": beta_values,
            "qubit": np.asarray(qubits, dtype=str),
            "fidelity": fidelity,
            "fidelity_std": fidelity_std,
            "error_per_gate": error_per_gate,
            "error_per_gate_std": error_per_gate_std,
            "success": success,
            "best_beta": best_beta,
            "best_fidelity": best_fidelity,
            "rows": rows,
        }

    def _plot_results(self) -> dict[str, Any]:
        beta = self.results["beta"]
        figures = {}
        for index, qubit in enumerate(self.results["qubit"]):
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.errorbar(
                beta,
                self.results["fidelity"][index],
                yerr=self.results["fidelity_std"][index],
                marker="o",
                capsize=3,
                label="RB fidelity",
            )
            if not np.isnan(self.results["best_beta"][index]):
                axis.axvline(
                    self.results["best_beta"][index],
                    color="tab:red",
                    linestyle="--",
                    label=f"best beta = {self.results['best_beta'][index]:.4g}",
                )
            axis.set_title(f"{qubit} DRAG beta sweep")
            axis.set_xlabel("DRAG beta")
            axis.set_ylabel("Single-qubit RB fidelity")
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figures[f"{qubit}_drag_sweep"] = figure
        plt.show()
        return figures

    def save_results(self) -> Path:
        """Save aggregate sweep arrays, metadata, and figures."""
        run_directory = self.saver.save(
            self.name,
            sweep={"beta": self.results["beta"], "qubit": self.results["qubit"]},
            results={
                "fidelity": self.results["fidelity"],
                "fidelity_std": self.results["fidelity_std"],
                "error_per_gate": self.results["error_per_gate"],
                "error_per_gate_std": self.results["error_per_gate_std"],
                "success": self.results["success"],
                "best_beta": self.results["best_beta"],
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
            "beta": self.results["beta"].tolist(),
            "best_beta": self.results["best_beta"].tolist(),
            "best_fidelity": self.results["best_fidelity"].tolist(),
            "fidelity_std": self.results["fidelity_std"].tolist(),
            "rows": self.results["rows"],
        }
        with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
        if self.results.get("figures"):
            self.saver.save_figures(run_directory, self.results["figures"])
        print(f"DRAG sweep saved to {run_directory}")
        return run_directory


def default_parameters() -> DragSweepParameters:
    """Return a practical DRAG sweep setup for a quick RB validation scan."""
    parameters = DragSweepParameters()
    parameters.beta_values = np.linspace(0.0, 2.0, 9)

    parameters.rb.use_state_discrimination = True
    parameters.rb.reset_type = "active"
    parameters.rb.gate_family = "drag"
    parameters.rb.max_circuit_depth = int(2**9)
    parameters.rb.delta_clifford = 4
    parameters.rb.num_random_sequences = 12
    parameters.rb.num_shots = 50
    parameters.rb.log_scale = True
    parameters.rb.use_strict_timing = True

    return parameters


if __name__ == "__main__":
    parameters = default_parameters()
    parameters.qubit = "q9"
    parameters.profile_name = "single_qubit"
    parameters.beta_values = np.linspace(0, 5, 50)
    parameters.save_results = True
    parameters.plot_results = True
    parameters.rb_options = CalibrationOptions(
        save_raw_data=False,
        save_figures=False,
        analyse_data=True,
        plot_data=False,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )

    drag_sweep = DragSweep(parameters)
    drag_sweep.run()
