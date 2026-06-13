import unittest

from qm.qua import declare, fixed, program

from utils.readout_macro import discriminate_i


class ReadoutMacroTests(unittest.TestCase):
    def test_builds_above_and_below_threshold_qua_expressions(self):
        with program():
            i_quadrature = declare(fixed)
            above = discriminate_i(i_quadrature, 0.1)
            below = discriminate_i(
                i_quadrature,
                0.1,
                state_1_when="below_threshold",
            )

        self.assertIsNotNone(above)
        self.assertIsNotNone(below)

    def test_rejects_unknown_threshold_direction(self):
        with program():
            i_quadrature = declare(fixed)
            with self.assertRaisesRegex(ValueError, "state_1_when"):
                discriminate_i(i_quadrature, 0.1, state_1_when="sideways")


if __name__ == "__main__":
    unittest.main()
