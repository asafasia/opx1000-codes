import unittest
from pathlib import Path

from qm.qua import declare, fixed, program
from quam_config import create_machine

from utils.readout_macro import (
    active_reset_configured,
    discriminate_i,
    discriminate_nearest_center,
    readout_state_configured,
)


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

    def test_builds_two_and_three_state_nearest_center_expressions(self):
        with program():
            i_quadrature = declare(fixed)
            q_quadrature = declare(fixed)
            ge_state = discriminate_nearest_center(
                i_quadrature,
                q_quadrature,
                [[-0.001, 0.0], [0.001, 0.0]],
            )
            gef_state = discriminate_nearest_center(
                i_quadrature,
                q_quadrature,
                [[-0.001, 0.0], [0.001, 0.0], [0.0, 0.001]],
            )

        self.assertIsNotNone(ge_state)
        self.assertIsNotNone(gef_state)

    def test_rejects_invalid_center_count_and_shape(self):
        with program():
            i_quadrature = declare(fixed)
            q_quadrature = declare(fixed)
            with self.assertRaisesRegex(ValueError, "num_states"):
                discriminate_nearest_center(
                    i_quadrature,
                    q_quadrature,
                    [[0.0, 0.0]],
                    num_states=1,
                )
            with self.assertRaisesRegex(ValueError, "at least 3"):
                discriminate_nearest_center(
                    i_quadrature,
                    q_quadrature,
                    [[0.0, 0.0], [0.1, 0.0]],
                    num_states=3,
                )
            with self.assertRaisesRegex(ValueError, "distance_scale"):
                discriminate_nearest_center(
                    i_quadrature,
                    q_quadrature,
                    [[0.0, 0.0], [0.1, 0.0]],
                    distance_scale=0,
                )

    def test_nearest_center_scales_deltas_before_squaring(self):
        source = (
            Path(__file__).parent.parent / "utils" / "readout_macro.py"
        ).read_text()

        self.assertIn("DEFAULT_DISTANCE_SCALE = 128.0", source)
        self.assertIn("(i_quadrature - i_center) * float(distance_scale)", source)
        self.assertIn("(q_quadrature - q_center) * float(distance_scale)", source)

    def test_configured_dispatch_uses_quam_two_and_three_state_methods(self):
        calls = []

        class Resonator:
            readout_discriminator = "quam"

        class Qubit:
            resonator = Resonator()

            def readout_state(self, state, pulse_name):
                calls.append((2, state, pulse_name))

            def readout_state_gef(self, state, pulse_name):
                calls.append((3, state, pulse_name))

        state = object()
        readout_state_configured(Qubit(), state, num_states=2)
        readout_state_configured(Qubit(), state, num_states=3)

        self.assertEqual(
            calls,
            [(2, state, "readout"), (3, state, "readout_GEF")],
        )

    def test_configured_dispatch_rejects_unknown_method(self):
        class Resonator:
            readout_discriminator = "unknown"

        class Qubit:
            resonator = Resonator()

        with self.assertRaisesRegex(ValueError, "discriminator"):
            readout_state_configured(Qubit(), object())

    def test_active_reset_builds_for_both_discriminators_and_state_counts(self):
        machine = create_machine(profile_name="single_qubit", qubit="q1")
        qubit = machine.qubits["q1"]

        for discriminator in ("quam", "nearest_center"):
            for num_states in (2, 3):
                with self.subTest(
                    discriminator=discriminator,
                    num_states=num_states,
                ):
                    with program():
                        state, attempts = active_reset_configured(
                            qubit,
                            num_states=num_states,
                            max_attempts=3,
                            pulse_name="readout",
                            discriminator=discriminator,
                        )
                    self.assertIsNotNone(state)
                    self.assertIsNotNone(attempts)

    def test_active_reset_validates_attempts_and_required_operations(self):
        class Resonator:
            readout_discriminator = "quam"

        class XY:
            operations = {"x180": object()}

        class Qubit:
            name = "q1"
            resonator = Resonator()
            xy = XY()

        with self.assertRaisesRegex(ValueError, "max_attempts"):
            active_reset_configured(Qubit(), max_attempts=0)
        with self.assertRaisesRegex(ValueError, "EF reset operation"):
            active_reset_configured(Qubit(), num_states=3)

    def test_active_reset_reinitializes_attempt_counter_per_macro_call(self):
        source = (
            Path(__file__).parent.parent / "utils" / "readout_macro.py"
        ).read_text()

        self.assertIn("attempts = declare(int)", source)
        self.assertIn("assign(attempts, 1)", source)
        self.assertNotIn("attempts = declare(int, value=1)", source)


if __name__ == "__main__":
    unittest.main()
