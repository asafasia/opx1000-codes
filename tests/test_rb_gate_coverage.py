import re
import unittest
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from quam_config.create_machine_from_profile import create_machine_from_profile
from quam_config.derived_gates import DERIVED_GATE_SPECS


class RBGateCoverageTests(unittest.TestCase):
    def test_generated_machine_contains_every_gate_used_by_rb(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations_v2"
            / "11a_single_qubit_randomized_benchmarking.py"
        ).read_text(encoding="utf-8")
        rb_gates = set(re.findall(r'qubit\.xy\.play\("([^"]+)"', source))
        machine = create_machine_from_profile("main", save=False)

        self.assertEqual(rb_gates, {"x180", "y180", "x90", "-x90", "y90", "-y90"})
        self.assertTrue(rb_gates.issubset(machine.qubits["q9"].xy.operations))

    def test_all_non_x180_rb_gates_are_centralized_derived_gates(self):
        self.assertEqual(
            set(DERIVED_GATE_SPECS),
            {"y180", "x90", "-x90", "y90", "-y90"},
        )

    def test_v2_rb_can_bind_logical_gates_to_drag_family(self):
        rb_module = import_module("calibrations_v2.11a_single_qubit_randomized_benchmarking")

        @dataclass
        class Pulse:
            length: int = 40
            amplitude: float = 0.2
            axis_angle: float = 0.0
            alpha: float = 0.7

        class XY:
            operations = {"x180_drag": Pulse()}

        class Qubit:
            name = "q_test"
            xy = XY()

        qubit = Qubit()
        rb_module.install_rb_gate_family([qubit], "drag")

        self.assertTrue(set(rb_module.RB_GATE_SPECS).issubset(qubit.xy.operations))
        self.assertAlmostEqual(qubit.xy.operations["x90"].amplitude, 0.1)
        self.assertAlmostEqual(qubit.xy.operations["y180"].axis_angle, rb_module.pi / 2)
        self.assertAlmostEqual(qubit.xy.operations["x90"].alpha, 0.7)
        self.assertIsNot(qubit.xy.operations["x180"], qubit.xy.operations["x180_drag"])

    def test_v2_rb_accepts_cos_alias(self):
        rb_module = import_module("calibrations_v2.11a_single_qubit_randomized_benchmarking")

        @dataclass
        class Pulse:
            length: int = 40
            amplitude: float = 0.2
            axis_angle: float = 0.0

        class XY:
            operations = {"x180_cosine": Pulse()}

        class Qubit:
            name = "q_test"
            xy = XY()

        qubit = Qubit()
        rb_module.install_rb_gate_family([qubit], "cos")

        self.assertAlmostEqual(qubit.xy.operations["x90"].amplitude, 0.1)

    def test_workflow_defaults_rb_to_drag_family(self):
        source = (
            Path(__file__).parent.parent / "workflows" / "drag_workflow.py"
        ).read_text(encoding="utf-8")

        self.assertIn('parameters.rb.gate_family = "drag"', source)


if __name__ == "__main__":
    unittest.main()
