import unittest
from pathlib import Path
from unittest.mock import patch

from profiles import Profile, ProfileError, clear_active_profile, current_profile_name, load_profile
from profiles.loader import _read_json
from quam_config import CreateMachine, create_machine
from quam_config.create_machine_from_profile import create_machine_from_profile
from quam_config.my_quam import apply_temporary_mw_fem_lo_mode_bugfix


class SingleQubitProfileTests(unittest.TestCase):
    def test_single_qubit_profile_is_complete_and_independent(self):
        profile_directory = Path(__file__).parent.parent / "profiles" / "single_qubit"

        self.assertEqual(
            {path.name for path in profile_directory.glob("*.json")},
            {"profile.json", "connectivity.json", "qubits.json", "pulses.json"},
        )
        manifest = _read_json(profile_directory / "profile.json")
        self.assertNotIn("base_profile", manifest)
        self.assertNotIn("lo_frequencies_file", manifest)

    def test_single_qubit_loader_reads_only_single_qubit_files(self):
        paths = []

        def record_read(path):
            paths.append(path)
            return _read_json(path)

        with patch("profiles.loader._read_json", side_effect=record_read):
            load_profile("single_qubit", qubit="q3")

        self.assertTrue(paths)
        self.assertTrue(all(path.parent.name == "single_qubit" for path in paths))

    def test_single_qubit_profile_contains_only_selected_qubit_and_ports(self):
        profile = load_profile("single_qubit", qubit="q3")
        connectivity = profile["connectivity"]
        fem = connectivity["controllers"]["con1"]["fems"]["7"]

        self.assertEqual(list(profile["qubits"]["qubits"]), ["q3"])
        self.assertEqual(list(profile["pulses"]["pulses"]), ["q3"])
        self.assertEqual(list(connectivity["connections"]), ["q3"])
        self.assertEqual(profile["manifest"]["active_qubits"], ["q3"])
        self.assertEqual(set(fem["outputs"]), {"1", "3"})
        self.assertEqual(set(fem["inputs"]), {"1"})
        self.assertEqual(fem["outputs"]["3"]["lo_frequency_hz"], 4_000_000_000)
        self.assertEqual(fem["outputs"]["1"]["lo_frequency_hz"], 6_900_000_000)
        self.assertEqual(
            connectivity["connections"]["q3"]["z_output"]["global_line"],
            "global_z",
        )
        self.assertEqual(set(connectivity["global_flux_lines"]), {"global_z"})

    def test_single_qubit_profile_can_be_loaded_without_selection_for_editing(self):
        profile = load_profile("single_qubit")

        self.assertIn("q3", profile["qubits"]["qubits"])
        self.assertNotIn("selected_qubit", profile["manifest"])
        self.assertEqual(
            profile["qubits"]["qubits"]["q3"]["flux"]["global_line"],
            "global_z",
        )

    def test_single_qubit_profile_rejects_unknown_qubit(self):
        with self.assertRaisesRegex(ProfileError, "does not exist in profile 'single_qubit'"):
            load_profile("single_qubit", qubit="q99")

    def test_single_qubit_profile_rejects_missing_selected_qubit_data(self):
        def read_without_q3_pulses(path):
            document = _read_json(path)
            if path.name == "pulses.json":
                document["pulses"].pop("q3")
            return document

        with patch("profiles.loader._read_json", side_effect=read_without_q3_pulses):
            with self.assertRaisesRegex(ProfileError, "has no pulse definitions"):
                load_profile("single_qubit", qubit="q3")

    def test_single_qubit_profile_rejects_missing_per_qubit_lo(self):
        def read_without_q3_lo(path):
            document = _read_json(path)
            if path.name == "connectivity.json":
                document["connections"]["q3"].pop("lo_frequencies_hz")
            return document

        with patch("profiles.loader._read_json", side_effect=read_without_q3_lo):
            with self.assertRaisesRegex(ProfileError, "has no lo_frequencies_hz"):
                load_profile("single_qubit", qubit="q3")

    def test_every_single_qubit_profile_qubit_can_be_selected(self):
        profile_directory = Path(__file__).parent.parent / "profiles" / "single_qubit"
        qubits = _read_json(profile_directory / "qubits.json")["qubits"]

        for qubit in qubits:
            with self.subTest(qubit=qubit):
                profile = load_profile("single_qubit", qubit=qubit)
                self.assertEqual(list(profile["qubits"]["qubits"]), [qubit])
                frequencies = profile["qubits"]["qubits"][qubit]["frequencies_hz"]
                connection = profile["connectivity"]["connections"][qubit]
                fem = profile["connectivity"]["controllers"]["con1"]["fems"]["7"]
                xy_lo = fem["outputs"][str(connection["xy_output"]["port"])][
                    "lo_frequency_hz"
                ]
                resonator_lo = fem["outputs"][
                    str(connection["resonator_output"]["port"])
                ]["lo_frequency_hz"]
                self.assertLessEqual(abs(frequencies["qubit_f01"] - xy_lo), 50_000_000)
                self.assertLessEqual(
                    abs(frequencies["resonator"] - resonator_lo),
                    50_000_000,
                )

    def test_single_qubit_profile_builds_one_qubit_machine(self):
        machine = create_machine_from_profile("single_qubit", save=False, qubit="q3")

        self.assertEqual(list(machine.qubits), ["q3"])
        self.assertEqual(machine.active_qubit_names, ["q3"])

    def test_external_global_flux_bias_is_kept_in_qubit_extras(self):
        machine = create_machine_from_profile("single_qubit", save=False, qubit="q10")
        qubit = machine.qubits["q10"]

        self.assertIsNone(qubit.z)
        self.assertEqual(qubit.extras["global_flux_line"], "global_z")
        self.assertEqual(qubit.extras["flux"]["flux_point"], "arbitrary")


class CreateMachineTests(unittest.TestCase):
    def tearDown(self):
        clear_active_profile()

    @patch("quam_config.create_machine_from_profile.create_machine_from_profile")
    def test_create_machine_class_returns_default_main_machine(self, build):
        expected = object()
        build.return_value = expected

        cm = CreateMachine()

        self.assertIs(cm.machine, expected)
        self.assertEqual(cm.profile.name, "main")
        self.assertEqual(current_profile_name(), "main")
        build.assert_called_once_with(cm.profile, save=False)

    @patch("quam_config.create_machine_from_profile.create_machine_from_profile")
    def test_legacy_positional_profile_selection_is_preserved(self, build):
        expected = object()
        build.return_value = expected

        self.assertIs(create_machine("testing"), expected)

        profile = build.call_args.args[0]
        self.assertIsInstance(profile, Profile)
        self.assertEqual(profile.name, "testing")
        build.assert_called_once_with(profile, save=False)

    @patch("quam_config.create_machine_from_profile.create_machine_from_profile")
    def test_qubit_selection_uses_single_qubit_profile(self, build):
        expected = object()
        build.return_value = expected

        self.assertIs(create_machine(qubit="q3"), expected)

        profile = build.call_args.args[0]
        self.assertEqual(profile.name, "single_qubit")
        self.assertEqual(profile.qubit, "q3")
        self.assertEqual(current_profile_name(), "single_qubit")
        build.assert_called_once_with(profile, save=False)

    @patch.dict("os.environ", {"QUAM_PROFILE": "single_qubit", "QUAM_QUBIT": "q5"})
    @patch("quam_config.create_machine_from_profile.create_machine_from_profile")
    def test_environment_can_override_script_qubit_selection(self, build):
        expected = object()
        build.return_value = expected

        self.assertIs(create_machine(qubit="q1"), expected)

        profile = build.call_args.args[0]
        self.assertEqual(profile.name, "single_qubit")
        self.assertEqual(profile.qubit, "q5")

    def test_main_mode_rejects_qubit_selection(self):
        with self.assertRaisesRegex(ProfileError, "does not support"):
            create_machine(mode="main", qubit="q3")

    def test_profile_object_can_be_passed_to_create_machine(self):
        with patch("quam_config.create_machine_from_profile.create_machine_from_profile") as build:
            expected = object()
            build.return_value = expected
            profile = Profile("single_qubit", qubit="q3")

            self.assertIs(create_machine(profile), expected)

            build.assert_called_once_with(profile, save=False)

    def test_temporary_mw_fem_lo_mode_bugfix_patches_generated_config(self):
        config = {
            "controllers": {
                "con1": {
                    "fems": {
                        7: {
                            "type": "MW",
                            "analog_inputs": {
                                1: {"band": 2, "downconverter_frequency": 5e9},
                                2: {"band": 2, "downconverter_frequency": 5.1e9},
                            },
                        }
                    }
                }
            }
        }

        apply_temporary_mw_fem_lo_mode_bugfix(config)

        inputs = config["controllers"]["con1"]["fems"][7]["analog_inputs"]
        self.assertEqual(inputs[1]["lo_mode"], "always_on")
        self.assertEqual(inputs[2]["lo_mode"], "always_on")


if __name__ == "__main__":
    unittest.main()
