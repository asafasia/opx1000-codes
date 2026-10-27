import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.profile_studio.server import list_profiles, read_section, write_section


class ProfileStudioTests(unittest.TestCase):
    def test_complete_repository_profiles_are_available(self):
        profiles = list_profiles()

        self.assertIn("Or_profile", profiles)
        self.assertNotIn("main", profiles)
        self.assertNotIn("single_qubit", profiles)
        self.assertIn("iqm_chip", profiles)

    def test_profiles_are_edited_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile_name, value in (("main", 1), ("iqm_chip", 2)):
                profile = root / profile_name
                profile.mkdir()
                for filename in ("profile.json", "qubits.json", "pulses.json", "connectivity.json"):
                    (profile / filename).write_text(
                        json.dumps({"value": value}) + "\n",
                        encoding="utf-8",
                    )

            with patch("apps.profile_studio.server.PROFILES_ROOT", root):
                section = read_section("iqm_chip", "qubits")
                write_section(
                    "iqm_chip",
                    "qubits",
                    {"value": 3},
                    section["digest"],
                )

            self.assertEqual(
                json.loads((root / "main" / "qubits.json").read_text())["value"],
                1,
            )
            self.assertEqual(
                json.loads((root / "iqm_chip" / "qubits.json").read_text())["value"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
