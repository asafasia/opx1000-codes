import re
import unittest
from pathlib import Path

from quam_config.create_machine_from_profile import create_machine_from_profile
from quam_config.derived_gates import DERIVED_GATE_SPECS


class RBGateCoverageTests(unittest.TestCase):
    def test_generated_machine_contains_every_gate_used_by_rb(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "11a_single_qubit_randomized_benchmarking.py"
        ).read_text()
        rb_gates = set(re.findall(r'qubit\.xy\.play\("([^"]+)"', source))
        machine = create_machine_from_profile("main", save=False)

        self.assertEqual(rb_gates, {"x180", "y180", "x90", "-x90", "y90", "-y90"})
        self.assertTrue(rb_gates.issubset(machine.qubits["q9"].xy.operations))

    def test_all_non_x180_rb_gates_are_centralized_derived_gates(self):
        self.assertEqual(
            set(DERIVED_GATE_SPECS),
            {"y180", "x90", "-x90", "y90", "-y90"},
        )


if __name__ == "__main__":
    unittest.main()
