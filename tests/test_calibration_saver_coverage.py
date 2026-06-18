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
                required += ("CalibrationSaver().save_figures(", "plot_per_qubit")
            absent = [item for item in required if item not in source]
            if absent:
                missing.append(f"{path.name}: {', '.join(absent)}")

        self.assertEqual(missing, [])

    def test_all_v2_xarray_saves_include_experiment_parameters(self):
        calibrations = Path(__file__).parent.parent / "calibrations_v2"
        missing = []
        for path in calibrations.glob("*.py"):
            source = path.read_text()
            search_from = 0
            while True:
                start = source.find("save_xarray(", search_from)
                if start == -1:
                    break
                depth = 0
                end = start
                for index, character in enumerate(source[start:], start=start):
                    if character == "(":
                        depth += 1
                    elif character == ")":
                        depth -= 1
                        if depth == 0:
                            end = index + 1
                            break
                call = source[start:end]
                if "parameters=" not in call:
                    line_number = source[:start].count("\n") + 1
                    missing.append(f"{path.name}:{line_number}")
                search_from = end

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
