import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profile_studio.server import list_profiles, read_section, write_section


class ProfileStudioTests(unittest.TestCase):
    def test_complete_repository_profiles_are_available(self):
        profiles = list_profiles()

        self.assertIn("main", profiles)
        self.assertIn("single_qubit", profiles)

    def test_profiles_are_edited_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile_name, value in (("main", 1), ("single_qubit", 2)):
                profile = root / profile_name
                profile.mkdir()
                for filename in ("profile.json", "qubits.json", "pulses.json", "connectivity.json"):
                    (profile / filename).write_text(
                        json.dumps({"value": value}) + "\n",
                        encoding="utf-8",
                    )

            with patch("profile_studio.server.PROFILES_ROOT", root):
                section = read_section("single_qubit", "qubits")
                write_section(
                    "single_qubit",
                    "qubits",
                    {"value": 3},
                    section["digest"],
                )

            self.assertEqual(
                json.loads((root / "main" / "qubits.json").read_text())["value"],
                1,
            )
            self.assertEqual(
                json.loads((root / "single_qubit" / "qubits.json").read_text())["value"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
