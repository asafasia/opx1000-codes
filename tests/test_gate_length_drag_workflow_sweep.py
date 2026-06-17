import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from calibration_io import CalibrationSaver
from sweeps.gate_length_drag_workflow_sweep import (
    GateLengthDragWorkflowSweep,
    GateLengthDragWorkflowSweepParameters,
    default_parameters,
    gate_lengths_from_range,
)
from workflows.drag_workflow import DragWorkflow, DragWorkflowParameters


def minimal_profile(length_ns=40):
    return {
        "manifest": {"build_mode": "multi_qubit", "active_qubits": ["q1"]},
        "connectivity": {"controllers": {}, "connections": {}},
        "qubits": {
            "qubits": {
                "q1": {
                    "operations": {
                        "x180": "x180",
                        "x180_drag": "x180_drag",
                        "x180_cosine": "x180_cosine",
                    }
                }
            }
        },
        "pulses": {
            "pulses": {
                "q1": {
                    "x180": {"type": "constant", "length_ns": length_ns},
                    "x180_drag": {
                        "type": "drag",
                        "length_ns": length_ns,
                        "beta": 0.0,
                    },
                    "x180_cosine": {
                        "type": "cosine",
                        "length_ns": length_ns,
                    },
                }
            }
        },
    }


class FakeWorkflow:
    seen_lengths = []

    def __init__(self, parameters):
        self.parameters = parameters
        self.calibrations = {}

    def run(self):
        length = self.parameters.gate_length_ns
        self.seen_lengths.append(length)
        error = abs(length - 26) / 1000
        self.calibrations["rb"] = SimpleNamespace(
            results={
                "fit_results": {
                    "q1": {
                        "success": True,
                        "error_per_gate": error,
                        "error_per_gate_std": 0.001,
                        "fidelity": 1.0 - error,
                        "fidelity_std": 0.001,
                    }
                }
            }
        )
        return {"rb": SimpleNamespace(mode="execute")}


class InterruptingFakeWorkflow(FakeWorkflow):
    completed_runs = 0

    def run(self):
        if self.completed_runs >= 1:
            raise KeyboardInterrupt
        self.__class__.completed_runs += 1
        return super().run()


class GateLengthDragWorkflowSweepTests(unittest.TestCase):
    def test_drag_workflow_applies_gate_length_to_selected_profile_pulses(self):
        parameters = DragWorkflowParameters(
            qubit="q1",
            profile_name="single_qubit",
            profile_documents=minimal_profile(),
            gate_length_ns=64,
            connect_before_run=False,
        )

        with patch("workflows.drag_workflow.validate_profile"):
            workflow = DragWorkflow(parameters)

        pulses = workflow.profile_documents["pulses"]["pulses"]["q1"]
        self.assertEqual(pulses["x180"]["length_ns"], 64)
        self.assertEqual(pulses["x180_drag"]["length_ns"], 64)
        self.assertEqual(pulses["x180_cosine"]["length_ns"], 64)
        self.assertEqual(
            workflow.profile_updates["gate_length"][
                "pulses.json.pulses.q1.x180_drag.length_ns"
            ],
            64,
        )

    def test_drag_workflow_rejects_non_multiple_of_4_gate_length(self):
        parameters = DragWorkflowParameters(
            qubit="q1",
            profile_name="single_qubit",
            profile_documents=minimal_profile(),
            gate_length_ns=66,
            connect_before_run=False,
        )

        with patch("workflows.drag_workflow.validate_profile"):
            with self.assertRaisesRegex(ValueError, "multiples of 4 ns"):
                DragWorkflow(parameters)

    def test_gate_length_sweep_runs_workflow_for_each_length_and_saves(self):
        FakeWorkflow.seen_lengths = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            parameters = GateLengthDragWorkflowSweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                profile_documents=minimal_profile(),
                gate_lengths_ns=np.asarray([8, 16, 28]),
                save_results=True,
                plot_results=True,
            )

            with patch("sweeps.gate_length_drag_workflow_sweep.validate_profile"), patch(
                "sweeps.gate_length_drag_workflow_sweep.plt.show"
            ):
                sweep = GateLengthDragWorkflowSweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                    workflow_class=FakeWorkflow,
                )
                results = sweep.run()

            self.assertEqual(FakeWorkflow.seen_lengths, [8, 16, 28])
            np.testing.assert_array_equal(results["gate_length_ns"], [8, 16, 28])
            np.testing.assert_allclose(results["best_gate_length_ns"], [28])
            self.assertTrue((sweep.run_directory / "summary.json").is_file())
            with np.load(sweep.run_directory / "results.npz") as saved:
                self.assertIn("fidelity_std", saved.files)
                self.assertIn("best_gate_length_ns", saved.files)

    def test_gate_length_sweep_saves_completed_points_on_keyboard_interrupt(self):
        FakeWorkflow.seen_lengths = []
        InterruptingFakeWorkflow.completed_runs = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            parameters = GateLengthDragWorkflowSweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                profile_documents=minimal_profile(),
                gate_lengths_ns=np.asarray([8, 16, 28]),
                save_results=True,
                plot_results=False,
            )

            with patch("sweeps.gate_length_drag_workflow_sweep.validate_profile"):
                sweep = GateLengthDragWorkflowSweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                    workflow_class=InterruptingFakeWorkflow,
                )
                with self.assertRaises(KeyboardInterrupt):
                    sweep.run()

            self.assertTrue((sweep.run_directory / "summary.json").is_file())
            summary = json.loads((sweep.run_directory / "summary.json").read_text())
            self.assertTrue(summary["interrupted"])
            self.assertEqual(summary["completed_points"], 1)
            self.assertEqual(summary["planned_points"], 3)
            with np.load(sweep.run_directory / "results.npz") as saved:
                self.assertTrue(bool(saved["interrupted"]))
                self.assertEqual(int(saved["completed_points"]), 1)

    def test_default_sweep_example_aligns_6_to_100_by_10_ns_to_qop_grid(self):
        parameters = default_parameters()

        np.testing.assert_array_equal(
            parameters.gate_lengths_ns,
            [8, 16, 28, 36, 48, 56, 68, 76, 88, 96],
        )
        self.assertEqual(parameters.workflow.rabi.operation, "x180_drag")
        self.assertEqual(parameters.workflow.rb.gate_family, "drag")

    def test_range_helper_can_align_requested_lengths(self):
        np.testing.assert_array_equal(
            gate_lengths_from_range(6, 100, 10),
            [8, 16, 28, 36, 48, 56, 68, 76, 88, 96],
        )

    def test_manual_invalid_gate_lengths_are_rejected_before_qop(self):
        parameters = GateLengthDragWorkflowSweepParameters(
            qubit="q1",
            profile_name="single_qubit",
            profile_documents=minimal_profile(),
            gate_lengths_ns=np.asarray([6, 16]),
            save_results=False,
            plot_results=False,
        )

        with patch("sweeps.gate_length_drag_workflow_sweep.validate_profile"):
            sweep = GateLengthDragWorkflowSweep(
                parameters,
                workflow_class=FakeWorkflow,
            )

        with self.assertRaisesRegex(ValueError, "multiples of 4 ns"):
            sweep.run()


if __name__ == "__main__":
    unittest.main()
