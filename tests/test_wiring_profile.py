import unittest
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

        self.assertEqual(profile["manifest"]["active_qubits"], ["q9"])
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

    def test_default_quam_load_builds_machine_from_profile(self):
        machine = object()

        with patch("quam_config.create_machine", return_value=machine) as create_machine:
            self.assertIs(Quam.load(), machine)

        create_machine.assert_called_once_with()

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
        readout = profile["qubits"]["qubits"]["q9"]["readout"]
        resonator = machine.qubits["q9"].resonator

        self.assertEqual(resonator.time_of_flight, readout["time_of_flight_ns"])
        self.assertEqual(resonator.smearing, readout["smearing_ns"])
        self.assertEqual(resonator.depletion_time, readout["depletion_time_ns"])

    def test_constant_readout_integration_weights_and_angle_are_applied(self):
        machine = create_machine_from_profile("main", save=False)
        profile = load_profile("main")
        readout = profile["qubits"]["qubits"]["q9"]["readout"]
        pulse_profile = profile["pulses"]["pulses"]["q9"]["readout"]
        pulse = machine.qubits["q9"].resonator.operations["readout"]

        self.assertEqual(pulse.integration_weights, pulse_profile["integration_weights"])
        self.assertEqual(
            pulse.integration_weights_angle,
            readout["integration_weights_angle_rad"],
        )
        self.assertEqual(pulse.threshold, readout["threshold"])
        self.assertEqual(pulse.rus_exit_threshold, readout["rus_exit_threshold"])


if __name__ == "__main__":
    unittest.main()
