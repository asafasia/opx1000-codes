import unittest
from pathlib import Path


class GEFReadoutFrequencySequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "14_gef_readout_frequency_optimization.py"
        ).read_text()

    def test_gef_readout_pulse_uses_fresh_integration_weights(self):
        self.assertIn("SquareReadoutPulse(", self.source)
        self.assertIn("integration_weights=[[1.0, new_length]]", self.source)
        self.assertNotIn("dataclasses.replace(", self.source)


if __name__ == "__main__":
    unittest.main()
