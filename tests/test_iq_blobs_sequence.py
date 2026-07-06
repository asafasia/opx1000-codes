import unittest
from pathlib import Path


class IQBlobsSequenceTests(unittest.TestCase):
    def test_preparation_uses_global_align_before_prepared_measurement(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        align_position = prepared_block.index("align()")
        measure_position = prepared_block.index("qubit.resonator.measure")
        self.assertIn("for i, qubit in multiplexed_qubits.items():", prepared_block[:align_position])
        self.assertIn("reset_qubit(qubit, i)", prepared_block[:align_position])
        self.assertLess(align_position, measure_position)
        self.assertNotIn("qubit.align()", prepared_block)

    def test_prepared_readout_has_no_extra_timing_delay(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        self.assertNotIn("xy_to_readout_delay_in_ns", prepared_block)
        self.assertNotIn("qubit.resonator.wait", prepared_block)

    def test_ground_and_prepared_clouds_use_independent_shot_loops(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()
        acquisition_block = source.split("# Acquire the selected clouds", 1)[1].split(
            "with stream_processing()", 1
        )[0]

        self.assertEqual(acquisition_block.count("with for_(n, 0, n < n_runs, n + 1):"), 3)
        self.assertIn('if "g" in states:', acquisition_block)
        self.assertIn('if "e" in states:', acquisition_block)
        self.assertIn('if "f" in states:', acquisition_block)
        self.assertNotIn("GEF_frequency_shift", acquisition_block)
        self.assertNotIn("qubit.resonator.wait", acquisition_block)

    def test_f_state_active_reset_uses_bounded_gef_reset(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()

        self.assertIn('use_gef_active_reset = "f" in states and reset_type == "active"', source)
        self.assertIn("active_gef_reset_attempts", source)
        self.assertIn("qubit.readout_state_gef(reset_state[qubit_index])", source)
        self.assertIn("reset_attempt < node.parameters.active_gef_reset_attempts", source)
        self.assertIn("with if_(reset_state[qubit_index] == 1):", source)
        self.assertIn("with if_(reset_state[qubit_index] == 2):", source)
        self.assertNotIn('qubit.reset(\n                        "active_gef"', source)
        self.assertIn('(operation == "readout_GEF" or use_gef_active_reset)', source)
        self.assertIn("length=readout_op.length", source)
        self.assertIn("integration_weights=_copy_integration_weights(", source)
        self.assertNotIn("deepcopy(readout_op.integration_weights)", source)
        self.assertNotIn("round(readout_op.length * 1.5 / 4)", source)
        self.assertIn("active_gef reset requires qubit.resonator.gef_centers", source)
        self.assertIn("Run IQ blobs with states ['g', 'e', 'f'] and reset_type='thermal' first.", source)

    def test_three_state_fit_updates_gef_centers(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()

        self.assertIn('state_labels == ["g", "e", "f"]', source)
        self.assertIn("q.resonator.gef_centers =", source)
        self.assertIn("centers * operation.length / 2**12", source)
        self.assertIn("readout.gef_centers", source)

    def test_successful_fit_updates_profile_angle_and_threshold(self):
        source = (Path(__file__).parent.parent / "calibrations_v2" / "07_iq_blobs.py").read_text()

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
