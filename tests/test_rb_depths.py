import unittest
from pathlib import Path

import numpy as np

from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters


class RbDepthTests(unittest.TestCase):
    def test_calibration_uses_for_each_for_arbitrary_depths(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations_v2"
            / "11a_single_qubit_randomized_benchmarking.py"
        ).read_text(encoding="utf-8")

        self.assertIn("with for_each_(depth, depths.tolist()):", source)
        self.assertNotIn("from_array(depth, depths)", source)

    def test_linear_depths_include_non_divisible_maximum(self):
        parameters = Parameters(
            log_scale=False,
            max_circuit_depth=204,
            delta_clifford=5,
        )

        depths = parameters.get_depths()

        np.testing.assert_array_equal(depths[:4], [1, 5, 10, 15])
        np.testing.assert_array_equal(depths[-2:], [200, 204])

    def test_linear_depths_do_not_duplicate_first_depth_for_unit_step(self):
        parameters = Parameters(
            log_scale=False,
            max_circuit_depth=4,
            delta_clifford=1,
        )

        np.testing.assert_array_equal(parameters.get_depths(), [1, 2, 3, 4])

    def test_log_depths_include_non_power_of_two_maximum(self):
        parameters = Parameters(log_scale=True, max_circuit_depth=204)

        np.testing.assert_array_equal(
            parameters.get_depths(),
            [1, 2, 4, 8, 16, 32, 64, 128, 204],
        )

    def test_invalid_depth_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_circuit_depth"):
            Parameters(max_circuit_depth=0).get_depths()
        with self.assertRaisesRegex(ValueError, "delta_clifford"):
            Parameters(delta_clifford=0).get_depths()


if __name__ == "__main__":
    unittest.main()
