import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from calibrations_v2.registry import CalibrationEntry, get_entry
from calibrations_v2.runner import (
    build_options,
    build_parameters,
    coerce_value,
    load_recipe,
    parse_assignment,
    run_entry,
)


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

    def test_load_recipe_reads_json_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text('{"calibration": "fake", "parameters": {"num_shots": 5}}\n', encoding="utf-8")

            self.assertEqual(load_recipe(path)["parameters"]["num_shots"], 5)

    def test_run_entry_instantiates_calibration_and_prints_summary(self):
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


if __name__ == "__main__":
    unittest.main()
