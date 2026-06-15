import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from profiles import Profile, clear_active_profile, set_active_profile
from updater import ProfileUpdater


class ProfileUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.profile = self.root / "profiles" / "main"
        self.profile.mkdir(parents=True)
        self.single_qubit_profile = self.root / "profiles" / "single_qubit"
        self.single_qubit_profile.mkdir(parents=True)
        (self.profile / "qubits.json").write_text(
            '{"qubits":{"q1":{"frequencies_hz":{"resonator":7470000000}}}}\n',
            encoding="utf-8",
        )
        (self.single_qubit_profile / "qubits.json").write_text(
            '{"qubits":{"q3":{"frequencies_hz":{"resonator":6875000000}}}}\n',
            encoding="utf-8",
        )
        self.updater = ProfileUpdater(
            self.root / "data" / "calibration_updates",
            self.root / "profiles",
        )

    def tearDown(self):
        clear_active_profile()
        self.temporary_directory.cleanup()

    def test_decline_keeps_profile_unchanged(self):
        proposal = self.updater.stage(
            "resonator_spectroscopy",
            {"qubits.json.qubits.q1.frequencies_hz.resonator": 7471000000},
            now=datetime(2026, 6, 12, tzinfo=timezone.utc),
        )
        with patch("builtins.input", return_value="no"):
            self.assertFalse(self.updater.confirm_and_apply(proposal))
        document = json.loads((self.profile / "qubits.json").read_text())
        self.assertEqual(document["qubits"]["q1"]["frequencies_hz"]["resonator"], 7470000000)

    def test_confirm_applies_and_backs_up(self):
        proposal = self.updater.stage(
            "resonator_spectroscopy",
            {"qubits.json.qubits.q1.frequencies_hz.resonator": 7471000000},
        )
        with patch("builtins.input", return_value="yes"):
            self.assertTrue(self.updater.confirm_and_apply(proposal))
        document = json.loads((self.profile / "qubits.json").read_text())
        self.assertEqual(document["qubits"]["q1"]["frequencies_hz"]["resonator"], 7471000000)
        self.assertTrue((proposal / "profile_before_update" / "qubits.json").is_file())
        self.assertEqual(json.loads((proposal / "proposal.json").read_text())["status"], "applied")

    def test_stage_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.updater.stage(
                "../outside",
                {"qubits.json.qubits.q1.frequencies_hz.resonator": 7471000000},
            )

    def test_stage_defaults_to_active_profile(self):
        set_active_profile(Profile("single_qubit", qubit="q3", root=self.root / "profiles"))

        proposal = self.updater.stage(
            "resonator_spectroscopy",
            {"qubits.json.qubits.q3.frequencies_hz.resonator": 6876000000},
        )

        document = json.loads((proposal / "proposal.json").read_text())
        self.assertEqual(document["profile_name"], "single_qubit")

    def test_stage_accepts_profile_object(self):
        proposal = self.updater.stage(
            "resonator_spectroscopy",
            {"qubits.json.qubits.q3.frequencies_hz.resonator": 6876000000},
            profile_name=Profile("single_qubit", qubit="q3", root=self.root / "profiles"),
        )

        document = json.loads((proposal / "proposal.json").read_text())
        self.assertEqual(document["profile_name"], "single_qubit")


if __name__ == "__main__":
    unittest.main()
