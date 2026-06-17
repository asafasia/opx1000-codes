import unittest
from pathlib import Path

from calibration_utils.power_rabi.parameters import Parameters, get_number_of_pulses


class PowerRabiOperationTests(unittest.TestCase):
    def test_x180_drag_is_an_allowed_pi_operation(self):
        parameters = Parameters(operation="x180_drag", max_number_pulses_per_sweep=6, pi_repetitions=1)

        self.assertEqual(get_number_of_pulses(parameters).tolist(), [1, 3, 5])

    def test_ef_transition_uses_pi_repetitions_for_ef_pi_pulses(self):
        parameters = Parameters(
            transition="ef",
            operation="x180_drag",
            max_number_pulses_per_sweep=6,
            pi_repetitions=3,
        )

        self.assertEqual(get_number_of_pulses(parameters).tolist(), [3])

    def test_sequence_accepts_configured_x180_variants(self):
        source = (
            Path(__file__).parent.parent / "calibrations" / "04b_power_rabi.py"
        ).read_text()

        self.assertIn('operation.startswith("x180_")', source)
        self.assertIn("ensure_operation_available(qubit, operation, node.parameters.transition)", source)


if __name__ == "__main__":
    unittest.main()
