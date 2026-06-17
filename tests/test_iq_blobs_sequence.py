import unittest
from pathlib import Path


class IQBlobsSequenceTests(unittest.TestCase):
    def test_preparation_uses_global_align_before_prepared_measurement(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        delay_position = prepared_block.index("node.parameters.xy_to_readout_delay_in_ns * u.ns")
        measure_position = prepared_block.index("qubit.resonator.measure")
        self.assertIn("align()", prepared_block[:delay_position])
        self.assertIn("for qubit in multiplexed_qubits.values():", prepared_block[:delay_position])
        self.assertLess(delay_position, measure_position)
        self.assertNotIn("qubit.align()", prepared_block)

    def test_explicit_delay_is_after_xy_alignment_and_before_readout(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]
        align_position = prepared_block.index("# Synchronize XY and resonator timelines")
        delay_position = prepared_block.index("node.parameters.xy_to_readout_delay_in_ns * u.ns")
        measure_position = prepared_block.index("qubit.resonator.measure")

        self.assertLess(align_position, delay_position)
        self.assertLess(delay_position, measure_position)

    def test_ground_and_prepared_clouds_use_independent_shot_loops(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        acquisition_block = source.split("# Acquire the ground and prepared clouds", 1)[1].split(
            "with stream_processing()", 1
        )[0]

        f_loop_marker = '\n            if "f" in states:\n                with for_(n, 0, n < n_runs, n + 1):'
        ge_block = acquisition_block.split(f_loop_marker, 1)[0]
        self.assertEqual(ge_block.count("with for_(n, 0, n < n_runs, n + 1):"), 2)
        self.assertIn(f_loop_marker, acquisition_block)
        self.assertEqual(acquisition_block.count("with for_(n, 0, n < n_runs, n + 1):"), 3)
        self.assertIn("qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)", acquisition_block)

    def test_successful_fit_updates_profile_angle_and_threshold(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()

        self.assertIn("readout.integration_weights_angle_rad", source)
        self.assertIn("readout.threshold", source)
        self.assertIn("readout.rus_exit_threshold", source)
        self.assertIn("proposing fitted parameters despite failed IQ-blob quality checks", source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", source)


if __name__ == "__main__":
    unittest.main()
