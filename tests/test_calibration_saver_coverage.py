import unittest
from pathlib import Path


class CalibrationSaverCoverageTests(unittest.TestCase):
    def test_all_acquisition_calibrations_use_calibration_saver(self):
        calibrations = Path(__file__).parent.parent / "calibrations"
        excluded = {"00_hello_qua.py"}

        missing = []
        for path in calibrations.glob("*.py"):
            source = path.read_text()
            if path.name in excluded or "def execute_qua_program" not in source:
                continue
            required = (
                "CalibrationSaver",
                "def save_raw_results",
                "CalibrationSaver().save_xarray(",
            )
            if "def plot_data" in source:
                required += ("CalibrationSaver().save_figures(",)
            absent = [item for item in required if item not in source]
            if absent:
                missing.append(f"{path.name}: {', '.join(absent)}")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
