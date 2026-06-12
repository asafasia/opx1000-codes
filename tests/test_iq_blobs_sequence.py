import unittest
from pathlib import Path


class IQBlobsSequenceTests(unittest.TestCase):
    def test_preparation_uses_global_align_before_prepared_measurement(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        prepared_block = source.split("with for_(n, 0, n < n_runs, n + 1):", 2)[2].split(
            "with stream_processing()", 1
        )[0]

        self.assertIn("align()\n                for i, qubit in multiplexed_qubits.items():", prepared_block)
        self.assertNotIn("qubit.align()", prepared_block)

    def test_ground_and_prepared_clouds_use_independent_shot_loops(self):
        source = (Path(__file__).parent.parent / "calibrations" / "07_iq_blobs.py").read_text()
        acquisition_block = source.split("# Acquire the ground and prepared clouds", 1)[1].split(
            "with stream_processing()", 1
        )[0]

        self.assertEqual(acquisition_block.count("with for_(n, 0, n < n_runs, n + 1):"), 2)
        self.assertIn("qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)", acquisition_block)


if __name__ == "__main__":
    unittest.main()
