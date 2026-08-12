import unittest
from importlib import import_module
from pathlib import Path

import numpy as np


class IQBlobsSequenceTests(unittest.TestCase):
    def test_center_rotation_matches_updated_integration_weight_frame(self):
        rotate_iq_centers = import_module(
            "calibrations.07_iq_blobs"
        )._rotate_iq_centers

        np.testing.assert_allclose(
            rotate_iq_centers([[1.0, 0.0], [0.0, 1.0]], np.pi / 2),
            [[0.0, 1.0], [-1.0, 0.0]],
            atol=1e-12,
        )

    def test_preparation_uses_global_align_before_prepared_measurement(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        align_position = prepared_block.index("align()")
        measure_position = prepared_block.index("measure_cloud(qubit")
        self.assertIn("for i, qubit in multiplexed_qubits.items():", prepared_block[:align_position])
        self.assertIn("reset_qubit(qubit, i)", prepared_block[:align_position])
        self.assertLess(align_position, measure_position)
        self.assertNotIn("qubit.align()", prepared_block)

    def test_prepared_readout_has_no_extra_timing_delay(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        self.assertNotIn("xy_to_readout_delay_in_ns", prepared_block)
        self.assertNotIn("qubit.resonator.wait", prepared_block)

    def test_ground_and_prepared_clouds_use_independent_shot_loops(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        acquisition_block = source.split("# Acquire the selected clouds", 1)[1].split(
            "with stream_processing()", 1
        )[0]

        self.assertEqual(acquisition_block.count("with for_(n, 0, n < n_runs, n + 1):"), 3)
        self.assertIn('if "g" in states:', acquisition_block)
        self.assertIn('if "e" in states:', acquisition_block)
        self.assertIn('if "f" in states:', acquisition_block)
        self.assertIn("measure_cloud(qubit", acquisition_block)
        self.assertNotIn("qubit.resonator.wait", acquisition_block)

    def test_readout_gef_clouds_use_the_gef_frequency_shift(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn('uses_gef_frequency = operation == "readout_GEF"', source)
        self.assertIn("+ qubit.resonator.GEF_frequency_shift", source)
        self.assertIn("qubit.resonator.update_frequency(", source)

    def test_f_state_active_reset_uses_bounded_gef_reset(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn('use_gef_active_reset = "f" in states and reset_type == "active"', source)
        self.assertIn("active_gef_reset_attempts", source)
        self.assertIn("qubit.readout_state_gef(reset_state[qubit_index])", source)
        self.assertIn("reset_attempt < node.parameters.active_gef_reset_attempts", source)
        self.assertIn("with if_(reset_state[qubit_index] == 1):", source)
        self.assertIn("with if_(reset_state[qubit_index] == 2):", source)
        self.assertNotIn('qubit.reset(\n                        "active_gef"', source)
        self.assertIn('or use_gef_active_reset', source)
        self.assertIn("length=readout_op.length", source)
        self.assertIn("integration_weights=_copy_integration_weights(", source)
        self.assertNotIn("deepcopy(readout_op.integration_weights)", source)
        self.assertNotIn("round(readout_op.length * 1.5 / 4)", source)
        self.assertIn("active_gef reset requires qubit.resonator.gef_centers", source)
        self.assertIn("Run IQ blobs with states ['g', 'e', 'f'] and reset_type='thermal' first.", source)

    def test_active_gef_uses_simple_repeat_until_ground_reset(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn("def reset_qubit_active_gef(qubit, max_attempts: int = 15)", source)
        self.assertIn("from utils.readout_macro import active_reset_configured", source)
        self.assertIn("active_reset_configured(", source)
        self.assertIn("num_states=3", source)
        self.assertIn("max_attempts=max_attempts", source)
        self.assertIn('reset_type == "active_gef"', source)
        self.assertIn("max_attempts=node.parameters.active_gef_reset_attempts", source)

    def test_two_and_three_state_fits_update_iq_centers(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn("if state_labels in (", source)
        self.assertNotIn("active-reset acquisitions must not replace", source)
        self.assertIn('state_labels == ["g", "e", "f"]', source)
        self.assertIn("centers = _rotate_iq_centers(", source)
        self.assertIn("q.resonator.gef_centers =", source)
        self.assertIn("centers * operation.length / 2**12", source)
        self.assertIn("readout.gef_centers", source)

    def test_successful_fit_updates_profile_angle_and_threshold(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn("readout.integration_weights_angle_rad", source)
        self.assertIn("readout.threshold", source)
        self.assertIn("readout.rus_exit_threshold", source)
        self.assertIn("proposing fitted parameters despite failed IQ-blob quality checks", source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", source)
        self.assertIn(
            'operation.integration_weights_angle -= float(fit_result["iw_angle"])',
            source,
        )
        self.assertNotIn(
            'operation.integration_weights_angle += float(fit_result["iw_angle"])',
            source,
        )


if __name__ == "__main__":
    unittest.main()
