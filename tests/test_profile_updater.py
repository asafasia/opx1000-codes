import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from updater import ProfileUpdater


class ProfileUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.profile = self.root / "profiles" / "main"
        self.profile.mkdir(parents=True)
        (self.profile / "qubits.json").write_text(
            '{"qubits":{"q1":{"frequencies_hz":{"resonator":7470000000}}}}\n',
            encoding="utf-8",
        )
        self.updater = ProfileUpdater(
            self.root / "data" / "calibration_updates",
            self.root / "profiles",
        )

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
