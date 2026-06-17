"""Power Rabi, DRAG, and RB calibration workflow."""

from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from calibrations_v2 import CalibrationOptions
from calibration_utils.drag_calibration_180_minus180 import Parameters as DragParameters
from calibration_utils.power_rabi import Parameters as PowerRabiParameters
from calibration_utils.single_qubit_randomized_benchmarking import (
    Parameters as RandomizedBenchmarkingParameters,
)
from profiles import Profile, validate_profile
from profiles.loader import _select_qubit
from quam_config import Quam
from quam_config.create_machine_from_profile import create_machine_from_profile

PowerRabi = importlib.import_module("calibrations_v2.04b_power_rabi").PowerRabi
DragCalibration180Minus180 = importlib.import_module(
    "calibrations_v2.10b_drag_calibration_180_minus_180"
).DragCalibration180Minus180
SingleQubitRandomizedBenchmarking = importlib.import_module(
    "calibrations_v2.11a_single_qubit_randomized_benchmarking"
).SingleQubitRandomizedBenchmarking


@dataclass
class DragWorkflowParameters:
    """Parameters and switches for the DRAG calibration workflow."""

    qubit: str = "q9"
    profile_name: str = "single_qubit"
    profile_documents: dict[str, Any] | None = None
    connect_before_run: bool = True
    close_existing_qms: bool = True
    gate_length_ns: int | None = None
    gate_length_operations: tuple[str, ...] = ("x180", "x180_drag", "x180_cosine")
    run_rabi: bool = True
    run_drag: bool = True
    run_rb: bool = True
    options: CalibrationOptions = field(
        default_factory=lambda: CalibrationOptions(
            save_raw_data=False,
            save_figures=False,
            update_state=False,
            propose_profile_update=False,
            apply_profile_update=False,
        )
    )
    rabi: PowerRabiParameters = field(default_factory=PowerRabiParameters)
    drag: DragParameters = field(default_factory=DragParameters)
    rb: RandomizedBenchmarkingParameters = field(
        default_factory=RandomizedBenchmarkingParameters
    )


class DragWorkflow:
    """Run the usual single-qubit gate tune-up sequence.

    Order:
    1. Power Rabi to calibrate the pulse amplitude.
    2. DRAG 180/-180 calibration to tune alpha.
    3. Single-qubit randomized benchmarking to validate the gate.
    """

    def __init__(
        self,
        parameters: DragWorkflowParameters | None = None,
        machine: Quam | None = None,
    ) -> None:
        self.parameters = parameters or DragWorkflowParameters()
        self.profile_documents = self._load_profile_documents()
        self.machine = machine
        self.results: dict[str, Any] = {}
        self.profile_updates: dict[str, dict[str, Any]] = {}
        self.calibration_options = self.parameters.options
        self.calibrations: dict[str, Any] = {}
        self._apply_requested_gate_length()

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

    def _build_machine(self) -> Quam:
        profile = InMemoryProfile(
            self.profile_documents,
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
        steps = []
        if self.parameters.run_rabi:
            steps.append(("rabi", PowerRabi, self.parameters.rabi))
        if self.parameters.run_drag:
            steps.append(("drag", DragCalibration180Minus180, self.parameters.drag))
        if self.parameters.run_rb:
            steps.append(("rb", SingleQubitRandomizedBenchmarking, self.parameters.rb))

        for name, calibration_class, calibration_parameters in steps:
            machine = self._build_machine()
            calibration = calibration_class(
                parameters=calibration_parameters,
                machine=machine,
                options=self.calibration_options,
            )
            print(f"\n=== Running {name}: {calibration.name} ===")
            self.results[name] = calibration.run()
            self.calibrations[name] = calibration
            updates = self._updates_from_calibration(name, calibration)
            self.profile_updates[name] = updates
            self._apply_profile_updates(updates)
        return self.results

    def _apply_requested_gate_length(self) -> None:
        if self.parameters.gate_length_ns is None:
            return
        length_ns = int(self.parameters.gate_length_ns)
        if length_ns <= 0:
            raise ValueError("gate_length_ns must be a positive integer.")
        if length_ns % 4 != 0:
            raise ValueError(
                f"gate_length_ns={length_ns} is invalid. QOP pulse lengths must be "
                "multiples of 4 ns. Use a value such as "
                f"{length_ns + (-length_ns % 4)} ns."
            )

        updates = {}
        for operation in self.parameters.gate_length_operations:
            path = self._pulse_update_path(operation, "length_ns", required=False)
            if path is not None:
                updates[path] = length_ns
        self._apply_profile_updates(updates)
        self.profile_updates["gate_length"] = updates

    def _pulse_update_path(
        self,
        operation: str,
        field: str,
        *,
        required: bool = True,
    ) -> str | None:
        qubit_profiles = self.profile_documents["qubits"]["qubits"]
        pulse_profiles = self.profile_documents["pulses"]["pulses"]
        qubit_profile = qubit_profiles[self.parameters.qubit]
        pulse_name = qubit_profile["operations"].get(operation)
        pulse_profile = (
            pulse_profiles[self.parameters.qubit].get(pulse_name)
            if pulse_name
            else None
        )
        if pulse_profile is None:
            if required:
                raise ValueError(
                    f"{self.parameters.qubit} operation {operation!r} has no profile pulse."
                )
            return None
        if field not in pulse_profile:
            if required:
                raise ValueError(
                    f"{self.parameters.qubit} pulse {pulse_name!r} has no {field!r} field."
                )
            return None
        return f"pulses.json.pulses.{self.parameters.qubit}.{pulse_name}.{field}"

    def _updates_from_calibration(self, name: str, calibration: Any) -> dict[str, Any]:
        if name == "rabi":
            return self._power_rabi_updates(calibration)
        if name == "drag":
            return self._drag_updates(calibration)
        if name == "rb":
            return self._rb_updates(calibration)
        return {}

    def _power_rabi_updates(self, calibration: Any) -> dict[str, Any]:
        updates = {}
        parameters = calibration.parameters
        operation = (
            "EF_x180"
            if getattr(parameters, "transition", "ge") == "ef"
            else parameters.operation
        )
        qubit_profiles = self.profile_documents["qubits"]["qubits"]
        for q in calibration.namespace.get("qubits", []):
            if calibration.outcomes.get(q.name) != "successful":
                continue
            pulse_name = qubit_profiles[q.name]["operations"].get(operation)
            if pulse_name is None:
                calibration.log(
                    f"In-memory update skipped: operation {operation!r} has no profile pulse."
                )
                continue
            updates[f"pulses.json.pulses.{q.name}.{pulse_name}.amplitude"] = float(
                calibration.results["fit_results"][q.name]["opt_amp"]
            )
        return updates

    def _drag_updates(self, calibration: Any) -> dict[str, Any]:
        updates = {}
        qubit_profiles = self.profile_documents["qubits"]["qubits"]
        pulse_profiles = self.profile_documents["pulses"]["pulses"]
        operation = calibration.parameters.operation
        for q in calibration.namespace.get("qubits", []):
            if calibration.outcomes.get(q.name) != "successful":
                continue
            pulse_name = qubit_profiles[q.name]["operations"].get(operation)
            pulse_profile = (
                pulse_profiles[q.name].get(pulse_name) if pulse_name else None
            )
            if pulse_profile is None or pulse_profile.get("type") != "drag":
                calibration.log(
                    f"In-memory update skipped: operation {operation!r} does not map "
                    f"to a DRAG pulse for {q.name}."
                )
                continue
            updates[f"pulses.json.pulses.{q.name}.{pulse_name}.beta"] = float(
                calibration.results["fit_results"][q.name]["alpha"]
            )
        return updates

    def _rb_updates(self, calibration: Any) -> dict[str, Any]:
        return {
            f"metrics.json.qubits.{q.name}.gates.single_qubit_average_fidelity": float(
                1 - calibration.results["fit_results"][q.name]["error_per_gate"]
            )
            for q in calibration.namespace.get("qubits", [])
            if calibration.outcomes.get(q.name) == "successful"
        }

    def _apply_profile_updates(self, updates: Mapping[str, Any]) -> None:
        if not updates:
            return
        for path, value in updates.items():
            _set_profile_path(self.profile_documents, path, value)
        validate_profile(self.profile_documents)


class InMemoryProfile(Profile):
    """Profile adapter whose load/save operations never touch profile files."""

    def __init__(
        self,
        documents: Mapping[str, Any],
        *,
        name: str,
        qubit: str | None = None,
    ) -> None:
        super().__init__(name, qubit=qubit)
        self.documents = deepcopy(documents)

    def for_qubit(self, qubit: str) -> "InMemoryProfile":
        return InMemoryProfile(self.documents, name=self.name, qubit=qubit)

    def load(self, *, qubit: str | None = None) -> dict[str, Any]:
        documents = deepcopy(self.documents)
        validate_profile(documents)
        return documents

    def save(self, documents: dict[str, Any] | None = None) -> None:
        self.documents = deepcopy(
            documents if documents is not None else self.documents
        )
        validate_profile(self.documents)


def _set_profile_path(profile_documents: dict[str, Any], path: str, value: Any) -> None:
    document_name, separator, nested_path = path.partition(".json.")
    if not separator:
        raise ValueError(f"Update path must look like 'file.json.field.path': {path!r}")
    document_key = "manifest" if document_name == "profile" else document_name
    keys = nested_path.split(".")
    target = profile_documents[document_key]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def default_parameters() -> DragWorkflowParameters:
    """Return a practical starting point mirroring the v2 examples."""
    parameters = DragWorkflowParameters()

    parameters.rabi.reset_type = "active"
    parameters.rabi.use_state_discrimination = True
    parameters.rabi.num_shots = 500
    parameters.rabi.transition = "ge"
    parameters.rabi.operation = "x180_drag"
    parameters.rabi.pi_repetitions = 4

    parameters.drag.use_state_discrimination = True
    parameters.drag.reset_type = "active"
    parameters.drag.num_shots = 100

    parameters.rb.use_state_discrimination = True
    parameters.rb.reset_type = "active"
    parameters.rb.gate_family = "drag"
    parameters.rb.max_circuit_depth = 1024
    parameters.rb.delta_clifford = 2
    parameters.rb.num_random_sequences = 30
    parameters.rb.num_shots = 100
    parameters.rb.log_scale = True
    parameters.rb.use_strict_timing = True

    return parameters


if __name__ == "__main__":
    parameters = default_parameters()
    parameters.qubit = "q9"
    parameters.options = CalibrationOptions(
        save_raw_data=False,
        save_figures=False,
        analyse_data=True,
        plot_data=True,
        update_state=False,
        propose_profile_update=False,
        apply_profile_update=False,
    )

    drag_workflow = DragWorkflow(parameters=parameters)
    drag_workflow.run()
