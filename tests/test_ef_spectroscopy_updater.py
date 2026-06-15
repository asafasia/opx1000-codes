import unittest
from pathlib import Path


class EFSpectroscopyUpdaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "12_Qubit_Spectroscopy_E_to_F.py"
        ).read_text()

    def test_stages_ef_frequency_and_anharmonicity(self):
        self.assertIn("from profiles import ProfileUpdater", self.source)
        self.assertIn("frequencies_hz.qubit_f12", self.source)
        self.assertIn("transmon.anharmonicity_hz", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)


if __name__ == "__main__":
    unittest.main()
