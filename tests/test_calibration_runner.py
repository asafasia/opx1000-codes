import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from calibrations.registry import CalibrationEntry, get_entry
from calibrations.runner import (
    build_options,
    build_parameters,
    coerce_value,
    load_recipe,
    parse_assignment,
    print_job_status,
    run_entry,
)
from calibrations.job_status import query_qop_status


class FakeParameters:
    simulate: bool = False
    load_data_id: str | None = None
    num_shots: int = 10
    enabled: bool = True


class FakeCalibration:
    def __init__(
        self,
        *,
        parameters,
        profile_name=None,
        qubit=None,
        options=None,
        auto_connect=False,
    ):
        self.parameters = parameters
        self.profile_name = profile_name
        self.qubit = qubit
        self.options = options
        self.auto_connect = auto_connect
        self.namespace = {"calibration_run_directory": Path("fake/run")}

    def run(self):
        return SimpleNamespace(
            name="fake",
            mode="simulate" if self.parameters.simulate else "execute",
            outcomes={"q1": "successful"},
            raw_data_saved=self.options.save_raw_data,
            figures_saved=self.options.save_figures,
            ai_review_saved=self.options.ai_review,
            profile_update_proposed=self.options.propose_profile_update,
        )


class FakeEntry(CalibrationEntry):
    def load_class(self):
        return FakeCalibration

    def load_parameters_class(self):
        return FakeParameters


class CalibrationRunnerTests(unittest.TestCase):
    def test_get_entry_accepts_friendly_and_module_stem_names(self):
        self.assertEqual(get_entry("power-rabi").class_name, "PowerRabi")
        self.assertEqual(get_entry("04b_power_rabi").key, "power-rabi")
        self.assertEqual(get_entry("cpmg").class_name, "CPMG")
        self.assertEqual(get_entry("17_cpmg").key, "cpmg")

    def test_coerce_value_parses_common_cli_types(self):
        self.assertIs(coerce_value("true"), True)
        self.assertIsNone(coerce_value("none"))
        self.assertEqual(coerce_value("3"), 3)
        self.assertEqual(coerce_value("0.25"), 0.25)
        self.assertEqual(coerce_value("q1"), "q1")

    def test_build_parameters_applies_overrides_and_mode_flags(self):
        parameters = build_parameters(
            FakeParameters,
            [parse_assignment("num_shots=25"), parse_assignment("enabled=false")],
            simulate=True,
            load_data_id="saved/run",
        )

        self.assertEqual(parameters.num_shots, 25)
        self.assertFalse(parameters.enabled)
        self.assertTrue(parameters.simulate)
        self.assertEqual(parameters.load_data_id, "saved/run")

    def test_build_options_defaults_to_no_apply(self):
        options = build_options([parse_assignment("analyse_data=false")])

        self.assertFalse(options.apply_profile_update)
        self.assertFalse(options.analyse_data)
        self.assertTrue(options.save_raw_data)

    def test_build_options_accepts_ai_review(self):
        options = build_options([parse_assignment("ai_review=true")])

        self.assertTrue(options.ai_review)

    def test_load_recipe_reads_json_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text('{"calibration": "fake", "parameters": {"num_shots": 5}}\n', encoding="utf-8")

            self.assertEqual(load_recipe(path)["parameters"]["num_shots"], 5)

    @patch("calibrations.runner.record_run_in_database", return_value=7)
    def test_run_entry_instantiates_calibration_and_prints_summary(self, record_mock):
        entry = FakeEntry("fake", "fake_module", "FakeCalibration")
        with patch("builtins.print") as print_mock:
            exit_code = run_entry(
                entry,
                parameter_assignments=[parse_assignment("num_shots=3")],
                option_assignments=[],
                profile_name="single_qubit",
                qubit="q1",
                simulate=True,
                load_data_id=None,
                apply=False,
                auto_connect=False,
                dry_run=False,
                no_save=False,
                no_plot=False,
            )

        self.assertEqual(exit_code, 0)
        printed = print_mock.call_args.args[0]
        self.assertIn('"mode": "simulate"', printed)
        self.assertIn('"run_directory": "fake\\\\run"', printed)
        self.assertIn('"ai_review_saved": false', printed)
        self.assertIn('"results_database_run_id": 7', printed)
        record_mock.assert_called_once()

    def test_query_qop_status_filters_active_jobs(self):
        qmm = SimpleNamespace(
            list_open_qms=lambda: ["qm-1"],
            get_jobs=lambda status=(): [
                SimpleNamespace(
                    id="job-1",
                    status=status[0],
                    description="power rabi",
                    is_simulation=False,
                )
            ],
        )

        status = query_qop_status(qmm)

        self.assertEqual(status.open_qms, ("qm-1",))
        self.assertTrue(status.has_active_jobs)
        self.assertEqual(status.jobs[0].id, "job-1")
        self.assertEqual(status.jobs[0].status, "Running")

    @patch("calibrations.runner.query_profile_qop_status")
    def test_print_job_status_reports_no_jobs(self, status_mock):
        status_mock.return_value = SimpleNamespace(
            open_qms=(),
            jobs=(),
            has_active_jobs=False,
        )
        with patch("builtins.print") as print_mock:
            print_job_status(
                profile_name="main",
                qubit=None,
                all_jobs=False,
                json_output=False,
            )

        printed = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("open_qms: 0", printed)
        self.assertIn("active_jobs: no", printed)
        self.assertIn("jobs: none", printed)

    @patch("calibrations.runner.query_profile_qop_status")
    def test_print_job_status_supports_json(self, status_mock):
        status_mock.return_value = SimpleNamespace(
            to_dict=lambda: {
                "open_qms": ["qm-1"],
                "has_active_jobs": True,
                "jobs": [{"id": "job-1", "status": "Running"}],
            }
        )
        with patch("builtins.print") as print_mock:
            print_job_status(
                profile_name=None,
                qubit="q3",
                all_jobs=True,
                json_output=True,
            )

        printed = print_mock.call_args.args[0]
        self.assertIn('"open_qms": [', printed)
        self.assertIn('"job-1"', printed)
        status_mock.assert_called_once_with(
            profile_name=None,
            qubit="q3",
            active_only=False,
        )


if __name__ == "__main__":
    unittest.main()
