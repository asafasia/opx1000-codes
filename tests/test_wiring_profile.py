import unittest
import csv
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from profiles import ProfileError, load_profile
from profiles.loader import validate_profile
from quam_config import Quam
from quam_config.create_machine_from_profile import create_machine_from_profile
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam
from quam_config.wiring_lffem_mwfem import _xy_lines


class WiringProfileTests(unittest.TestCase):
    def test_active_qubits_are_defined_only_in_manifest(self):
        profile = load_profile("main")

        self.assertTrue(profile["manifest"]["active_qubits"])
        for qubit_name in profile["manifest"]["active_qubits"]:
            self.assertIn(qubit_name, profile["qubits"]["qubits"])
        for qubit in profile["qubits"]["qubits"].values():
            self.assertNotIn("enabled", qubit)

    def test_profile_rejects_per_qubit_enabled_field(self):
        profile = deepcopy(load_profile("main"))
        profile["qubits"]["qubits"]["q9"]["enabled"] = True

        with self.assertRaisesRegex(ProfileError, "use profile.json active_qubits"):
            validate_profile(profile)

    def test_xy_lines_group_qubits_on_the_same_output(self):
        connections = {
            "q9": {"xy_output": {"controller": "con1", "fem": 7, "port": 2}},
            "q10": {"xy_output": {"controller": "con1", "fem": 7, "port": 2}},
        }

        lines = _xy_lines(connections, ["q9", "q10"])

        self.assertEqual(lines, {(1, 7, 2): [9, 10]})

    def test_iqm_summary_keeps_vendor_q7_q8_frequencies(self):
        csv_path = Path(__file__).parent.parent / "docs" / "hardware" / "iqm_qubit_summary.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = {row["Qubit"]: row for row in csv.DictReader(file)}

        self.assertEqual(float(rows["7"]["Measured max. qubit frequency (GHz)"]), 3.9366)
        self.assertEqual(float(rows["7"]["Measured readout frequency (GHz)"]), 7.2712)
        self.assertEqual(float(rows["8"]["Measured max. qubit frequency (GHz)"]), 4.098)
        self.assertEqual(float(rows["8"]["Measured readout frequency (GHz)"]), 7.3834)

    def test_new_qubits_use_vendor_table_frequencies(self):
        profile = load_profile("main")
        qubits = profile["qubits"]["qubits"]
        csv_path = Path(__file__).parent.parent / "docs" / "hardware" / "iqm_qubit_summary.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = {row["Qubit"]: row for row in csv.DictReader(file)}

        for index in range(1, 7):
            settings = qubits[f"q{index}"]
            self.assertAlmostEqual(
                settings["frequencies_hz"]["qubit_f01"],
                float(rows[str(index)]["Measured max. qubit frequency (GHz)"]) * 1e9,
                delta=1,
            )
            self.assertAlmostEqual(
                settings["frequencies_hz"]["resonator"],
                float(rows[str(index)]["Measured readout frequency (GHz)"]) * 1e9,
                delta=1,
            )

    def test_all_configured_qubits_use_requested_shared_drive_ports(self):
        profile = load_profile("main")
        connections = profile["connectivity"]["connections"]
        qubit_names = [f"q{index}" for index in range(1, 11)]
        lines = _xy_lines(connections, qubit_names)

        for port, first_qubit in enumerate(range(1, 10, 2), start=2):
            self.assertEqual(lines[(1, 7, port)], [first_qubit, first_qubit + 1])

    def test_all_drive_lines_use_same_lo(self):
        outputs = load_profile("main")["connectivity"]["controllers"]["con1"]["fems"]["7"]["outputs"]

        for port in range(3, 7):
            self.assertEqual(outputs["2"]["lo_frequency_hz"], outputs[str(port)]["lo_frequency_hz"])
            self.assertEqual(outputs["2"]["band"], outputs[str(port)]["band"])

    def test_profile_contains_q1_through_q10_only(self):
        profile = load_profile("main")

        expected = {f"q{index}" for index in range(1, 11)}
        self.assertEqual(set(profile["qubits"]["qubits"]), expected)
        self.assertEqual(set(profile["connectivity"]["connections"]), expected)
        self.assertEqual(set(profile["pulses"]["pulses"]), expected)

    def test_profile_rejects_different_los_on_shared_output_pair(self):
        profile = deepcopy(load_profile("main"))
        outputs = profile["connectivity"]["controllers"]["con1"]["fems"]["7"]["outputs"]
        outputs["3"]["lo_frequency_hz"] = outputs["2"]["lo_frequency_hz"] + 100e6

        with self.assertRaisesRegex(ProfileError, "outputs 2 and 3 share an LO"):
            validate_profile(profile)

    def test_hardware_reference_files_are_documented(self):
        hardware_docs = Path(__file__).parent.parent / "docs" / "hardware"

        self.assertTrue((hardware_docs / "wiring.md").is_file())
        self.assertTrue((hardware_docs / "iqm_qubit_summary.csv").is_file())

    def test_default_quam_load_builds_machine_from_profile(self):
        machine = object()

        with patch("quam_config.create_machine", return_value=machine) as create_machine:
            self.assertIs(Quam.load(), machine)

        create_machine.assert_called_once_with()

    def test_machine_wiring_contains_all_configured_qubits(self):
        machine = create_machine_from_profile("main", save=False)

        self.assertEqual(set(machine.qubits), {f"q{index}" for index in range(1, 11)})
        self.assertEqual(machine.active_qubit_names, ["q10"])

    def test_quam_save_skips_implicit_root_state(self):
        machine = Quam()

        with patch.object(FluxTunableQuam, "save") as upstream_save:
            machine.save()
            machine.save(Path("."))
            machine.save(Path("snapshot"))

        upstream_save.assert_called_once_with(Path("snapshot"), None, None, None)

    def test_calibrations_do_not_load_generated_root_state(self):
        calibrations = Path(__file__).parent.parent / "calibrations"

        for calibration in calibrations.glob("*.py"):
            with self.subTest(calibration=calibration.name):
                self.assertNotIn("Quam.load()", calibration.read_text(encoding="utf-8"))

    def test_readout_acquisition_timing_is_applied_to_quam(self):
        machine = create_machine_from_profile("main", save=False)
        profile = load_profile("main")
        qubit_name = profile["manifest"]["active_qubits"][0]
        readout = profile["qubits"]["qubits"][qubit_name]["readout"]
        resonator = machine.qubits[qubit_name].resonator

        self.assertEqual(resonator.time_of_flight, readout["time_of_flight_ns"])
        self.assertEqual(resonator.smearing, readout["smearing_ns"])
        self.assertEqual(resonator.depletion_time, readout["depletion_time_ns"])

    def test_constant_readout_integration_weights_and_angle_are_applied(self):
        machine = create_machine_from_profile("main", save=False)
        profile = load_profile("main")
        qubit_name = profile["manifest"]["active_qubits"][0]
        readout = profile["qubits"]["qubits"][qubit_name]["readout"]
        pulse_profile = profile["pulses"]["pulses"][qubit_name]["readout"]
        pulse = machine.qubits[qubit_name].resonator.operations["readout"]

        self.assertEqual(pulse.integration_weights, pulse_profile["integration_weights"])
        self.assertEqual(
            pulse.integration_weights_angle,
            readout["integration_weights_angle_rad"],
        )
        self.assertEqual(pulse.threshold, readout["threshold"])
        self.assertEqual(pulse.rus_exit_threshold, readout["rus_exit_threshold"])

    def test_rb_gates_are_derived_from_x180(self):
        machine = create_machine_from_profile("main", save=False)
        qubit = machine.qubits[machine.active_qubit_names[0]]
        x180 = qubit.xy.operations["x180"]
        expected = {
            "y180": (x180.amplitude, 0.5 * 3.141592653589793),
            "x90": (x180.amplitude / 2, 0.0),
            "-x90": (-x180.amplitude / 2, 0.0),
            "y90": (x180.amplitude / 2, 0.5 * 3.141592653589793),
            "-y90": (-x180.amplitude / 2, 0.5 * 3.141592653589793),
        }

        for gate_name, (amplitude, axis_angle) in expected.items():
            gate = qubit.xy.operations[gate_name]
            self.assertEqual(gate.length, x180.length)
            self.assertAlmostEqual(gate.amplitude, amplitude)
            self.assertAlmostEqual(gate.axis_angle, axis_angle)

    def test_profile_does_not_duplicate_derived_rb_gates(self):
        profile = load_profile("main")
        derived_gates = {"y180", "x90", "-x90", "y90", "-y90"}

        for qubit_name, qubit in profile["qubits"]["qubits"].items():
            self.assertTrue(derived_gates.isdisjoint(qubit["operations"]))
            self.assertTrue(derived_gates.isdisjoint(profile["pulses"]["pulses"][qubit_name]))


if __name__ == "__main__":
    unittest.main()
