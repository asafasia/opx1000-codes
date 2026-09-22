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



if __name__ == "__main__":
    unittest.main()
