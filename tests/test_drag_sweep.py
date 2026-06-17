import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from calibration_io import CalibrationSaver
from calibrations_v2 import CalibrationOptions
from sweeps.drag_sweep import DragSweep, DragSweepParameters


def minimal_profile(beta=0.0):
    return {
        "manifest": {"build_mode": "multi_qubit", "active_qubits": ["q1"]},
        "connectivity": {
            "controllers": {},
            "connections": {},
        },
        "qubits": {
            "qubits": {
                "q1": {
                    "operations": {
                        "x180_drag": "x180_drag",
                    }
                }
            }
        },
        "pulses": {
            "pulses": {
                "q1": {
                    "x180_drag": {
                        "type": "drag",
                        "beta": beta,
                    }
                }
            }
        },
    }


class FakeRB:
    seen_betas = []
    seen_gate_families = []

    def __init__(self, *, parameters, machine, options):
        self.parameters = parameters
        self.machine = machine
        self.options = options
        self.results = {}
        self.outcomes = {}

    def run(self):
        beta = self.machine["beta"]
        self.seen_betas.append(beta)
        self.seen_gate_families.append(self.parameters.gate_family)
        self.results["fit_results"] = {
            "q1": {
                "success": True,
                "error_per_gate": abs(beta - 0.5) / 10,
                "error_per_gate_std": 0.001 + beta / 1000,
                "fidelity_std": 0.001 + beta / 1000,
            }
        }
        self.outcomes = {"q1": "successful"}
        return SimpleNamespace(mode="execute")


class InterruptingFakeRB(FakeRB):
    completed_runs = 0

    def run(self):
        if self.completed_runs >= 1:
            raise KeyboardInterrupt
        self.__class__.completed_runs += 1
        return super().run()


class DragSweepTests(unittest.TestCase):
    def test_drag_sweep_runs_rb_for_each_beta_and_saves_summary(self):
        FakeRB.seen_betas = []
        FakeRB.seen_gate_families = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            parameters = DragSweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                profile_documents=minimal_profile(),
                beta_values=np.asarray([0.0, 0.5, 1.0]),
                connect_before_run=False,
                save_results=True,
                plot_results=True,
                rb_options=CalibrationOptions(plot_data=False),
            )
            with patch("sweeps.drag_sweep.validate_profile"), patch(
                "sweeps.drag_sweep.create_machine_from_profile",
                side_effect=lambda profile, save=False: {
                    "beta": profile.documents["pulses"]["pulses"]["q1"]["x180_drag"]["beta"]
                },
            ), patch("sweeps.drag_sweep.SingleQubitRandomizedBenchmarking", FakeRB), patch(
                "sweeps.drag_sweep.plt.show"
            ):
                sweep = DragSweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                )
                results = sweep.run()

            np.testing.assert_allclose(FakeRB.seen_betas, [0.0, 0.5, 1.0])
            self.assertEqual(FakeRB.seen_gate_families, ["drag", "drag", "drag"])
            np.testing.assert_allclose(results["best_beta"], [0.5])
            np.testing.assert_allclose(results["fidelity_std"], [[0.001, 0.0015, 0.002]])
            self.assertTrue((sweep.run_directory / "summary.json").is_file())
            self.assertTrue((sweep.run_directory / "figures" / "q1_drag_sweep.png").is_file())
            with np.load(sweep.run_directory / "results.npz") as saved:
                self.assertIn("fidelity_std", saved.files)
                self.assertIn("error_per_gate_std", saved.files)

    def test_drag_sweep_saves_completed_points_on_keyboard_interrupt(self):
        FakeRB.seen_betas = []
        FakeRB.seen_gate_families = []
        InterruptingFakeRB.completed_runs = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            parameters = DragSweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                profile_documents=minimal_profile(),
                beta_values=np.asarray([0.0, 0.5, 1.0]),
                connect_before_run=False,
                save_results=True,
                plot_results=False,
                rb_options=CalibrationOptions(plot_data=False),
            )
            with patch("sweeps.drag_sweep.validate_profile"), patch(
                "sweeps.drag_sweep.create_machine_from_profile",
                side_effect=lambda profile, save=False: {
                    "beta": profile.documents["pulses"]["pulses"]["q1"]["x180_drag"]["beta"]
                },
            ), patch(
                "sweeps.drag_sweep.SingleQubitRandomizedBenchmarking",
                InterruptingFakeRB,
            ):
                sweep = DragSweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
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


if __name__ == "__main__":
    unittest.main()
