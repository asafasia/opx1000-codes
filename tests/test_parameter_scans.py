import csv
import json
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parameter_scans.runner import ExperimentSpec, LongScanRunner, ScanConfig, extract_fit_records


class ParameterScanTests(unittest.TestCase):
    def test_extract_fit_records_keeps_only_numeric_fit_parameters(self):
        node = SimpleNamespace(
            results={
                "fit_results": {
                    "q1": {
                        "t1": 12000.0,
                        "t1_error": 500.0,
                        "success": True,
                        "note": "ignored",
                    }
                }
            },
            outcomes={},
        )

        records = extract_fit_records(
            node,
            timestamp="2026-06-15T12:00:00+03:00",
            cycle=2,
            experiment_name="05_T1",
            script=Path("calibrations/05_T1.py"),
            duration_s=1.25,
        )

        self.assertEqual([record["parameter"] for record in records], ["T1", "T1 error"])
        self.assertEqual(records[0]["unit"], "ns")
        self.assertEqual(records[0]["qubit"], "q1")
        self.assertTrue(records[0]["success"])

    def test_runner_saves_summary_after_successful_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_t1.py"
            script.write_text(
                "from types import SimpleNamespace\n"
                "node = SimpleNamespace(\n"
                "    name='05_T1',\n"
                "    results={'fit_results': {'q2': {'t1': 23000.0, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=1,
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            with (run_directory / "summary.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["parameter"], "T1")
            self.assertEqual(rows[0]["value"], "23000.0")
            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")

    def test_runner_records_failure_and_stops_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "broken.py"
            script.write_text("raise RuntimeError('hardware unhappy')\n", encoding="utf-8")
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=5,
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            with (run_directory / "summary.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "error")
            self.assertIn("hardware unhappy", rows[0]["error"])
            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")

    def test_runner_suppresses_full_calibration_saves_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class FakeCalibrationSaver:
                def save(self, experiment_name, sweep, results, **kwargs):
                    raise AssertionError("raw save should be patched")

                def save_xarray(self, experiment_name, dataset, **kwargs):
                    raise AssertionError("raw save should be patched")

                def save_figures(self, run_directory, figures):
                    raise AssertionError("figure save should be patched")

            fake_package = types.ModuleType("calibration_io")
            fake_saver_module = types.ModuleType("calibration_io.calibration_saver")
            fake_package.__path__ = []
            fake_package.CalibrationSaver = FakeCalibrationSaver
            fake_saver_module.CalibrationSaver = FakeCalibrationSaver
            script = root / "fake_save.py"
            script.write_text(
                "from types import SimpleNamespace\n"
                "from calibration_io import CalibrationSaver\n"
                "run_directory = CalibrationSaver().save_xarray('05_T1', object())\n"
                "CalibrationSaver().save_figures(run_directory, {'q1': object()})\n"
                "node = SimpleNamespace(\n"
                "    name='05_T1',\n"
                "    results={'fit_results': {'q1': {'t1': 12000.0, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=1,
                output_root=root / "data" / "parameter_scans",
            )

            with patch.dict(
                "sys.modules",
                {
                    "calibration_io": fake_package,
                    "calibration_io.calibration_saver": fake_saver_module,
                },
            ):
                run_directory = LongScanRunner(config, repository_root=root).run()

            self.assertTrue((run_directory / "skipped_raw_calibration_saves").is_dir())
            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")

    def test_runner_sets_profile_and_qubit_environment_only_during_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "env_probe.py"
            script.write_text(
                "import os\n"
                "from types import SimpleNamespace\n"
                "node = SimpleNamespace(\n"
                "    name='env_probe',\n"
                "    results={'fit_results': {'q7': {'frequency': 4.2e9, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n"
                "assert os.environ['QUAM_PROFILE'] == 'single_qubit'\n"
                "assert os.environ['QUAM_QUBIT'] == 'q7'\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=1,
                profile_name="single_qubit",
                qubit="q7",
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")

    def test_runner_extracts_results_from_v2_calibration_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_v2.py"
            script.write_text(
                "from types import SimpleNamespace\n"
                "calibration = SimpleNamespace(\n"
                "    name='05_T1',\n"
                "    results={'fit_results': {'q2': {'t1': 18000.0, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=1,
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            with (run_directory / "summary.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_name"], "05_T1")
            self.assertEqual(rows[0]["parameter"], "T1")

    def test_runner_resolves_legacy_calibrations_path_to_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2_directory = root / "calibrations_v2"
            v2_directory.mkdir()
            (v2_directory / "05_T1.py").write_text(
                "from types import SimpleNamespace\n"
                "calibration = SimpleNamespace(\n"
                "    name='05_T1',\n"
                "    results={'fit_results': {'q1': {'t1': 12000.0, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=Path("calibrations/05_T1.py"))],
                repetitions=1,
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")

    def test_runner_closes_plots_and_suppresses_show_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "plotting_script.py"
            script.write_text(
                "import matplotlib.pyplot as plt\n"
                "from types import SimpleNamespace\n"
                "plt.figure()\n"
                "plt.plot([0, 1], [0, 1])\n"
                "plt.show()\n"
                "node = SimpleNamespace(\n"
                "    name='plotting_script',\n"
                "    results={'fit_results': {'q1': {'frequency': 4.2e9, 'success': True}}},\n"
                "    outcomes={},\n"
                ")\n",
                encoding="utf-8",
            )
            config = ScanConfig(
                name="test_scan",
                experiments=[ExperimentSpec(script=script)],
                repetitions=1,
                output_root=root / "data" / "parameter_scans",
            )

            run_directory = LongScanRunner(config, repository_root=root).run()

            import matplotlib.pyplot as plt

            status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
