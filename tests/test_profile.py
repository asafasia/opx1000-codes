import json
import tempfile
import unittest
from pathlib import Path

from profiles import Profile, ProfileError


class ProfileTests(unittest.TestCase):
    def _write_profile(self, root: Path, name: str = "main") -> None:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "profile.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": name,
                    "files": {
                        "connectivity": "connectivity.json",
                        "qubits": "qubits.json",
                        "pulses": "pulses.json",
                    },
                    "active_qubits": ["q1"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "connectivity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "network": {"host": "127.0.0.1", "port": None, "cluster_name": "test"},
                    "controllers": {
                        "con1": {
                            "type": "opx1000",
                            "fems": {
                                "7": {
                                    "type": "mw_fem",
                                    "outputs": {
                                        "1": {
                                            "band": 3,
                                            "lo_frequency_hz": 7000000000,
                                            "full_scale_power_dbm": -10,
                                            "sampling_rate_hz": 1000000000,
                                        },
                                        "2": {
                                            "band": 1,
                                            "lo_frequency_hz": 4000000000,
                                            "full_scale_power_dbm": -10,
                                            "sampling_rate_hz": 1000000000,
                                        },
                                    },
                                    "inputs": {
                                        "1": {"band": 3, "sampling_rate_hz": 1000000000}
                                    },
                                }
                            },
                        }
                    },
                    "connections": {
                        "q1": {
                            "xy_output": {"controller": "con1", "fem": 7, "port": 2},
                            "resonator_output": {"controller": "con1", "fem": 7, "port": 1},
                            "resonator_input": {"controller": "con1", "fem": 7, "port": 1},
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "qubits.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "qubits": {
                        "q1": {
                            "grid_location": [0, 0],
                            "frequencies_hz": {
                                "qubit_f01": 4000000000,
                                "qubit_f12": None,
                                "resonator": 7000000000,
                                "resonator_bare": None,
                            },
                            "transmon": {
                                "anharmonicity_hz": 150000000,
                                "t1_ns": 10000,
                                "t2_ramsey_ns": None,
                                "t2_echo_ns": None,
                                "thermalization_time_ns": 100000,
                            },
                            "readout": {
                                "time_of_flight_ns": 320,
                                "smearing_ns": 0,
                                "depletion_time_ns": 2500,
                                "threshold": 0,
                                "rus_exit_threshold": 0,
                                "state_1_when": "above_threshold",
                                "integration_weights_angle_rad": 0,
                            },
                            "operations": {"x180": "x180_const", "readout": "readout"},
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "pulses.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pulses": {
                        "q1": {
                            "x180_const": {
                                "target": "qubit",
                                "type": "constant",
                                "amplitude": 0.1,
                                "length_ns": 40,
                            },
                            "readout": {
                                "target": "resonator",
                                "type": "constant",
                                "amplitude": 0.1,
                                "length_ns": 1000,
                                "integration_weights": [[1, 1000]],
                            },
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_profile_save_writes_complete_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            profile = Profile("main", root=root)
            documents = profile.load()
            documents["qubits"]["qubits"]["q1"]["readout"]["threshold"] = 0.25

            profile.save(documents)

            saved = json.loads((root / "main" / "qubits.json").read_text())
            self.assertEqual(saved["qubits"]["q1"]["readout"]["threshold"], 0.25)

    def test_profile_save_rejects_selected_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root, name="single_qubit")
            manifest = json.loads((root / "single_qubit" / "profile.json").read_text())
            manifest["build_mode"] = "single_qubit"
            (root / "single_qubit" / "profile.json").write_text(json.dumps(manifest) + "\n")
            connectivity = json.loads((root / "single_qubit" / "connectivity.json").read_text())
            connectivity["connections"]["q1"]["lo_frequencies_hz"] = {
                "xy_output": 4000000000,
                "resonator_output": 7000000000,
            }
            (root / "single_qubit" / "connectivity.json").write_text(json.dumps(connectivity) + "\n")
            profile = Profile("single_qubit", qubit="q1", root=root)
            selected = profile.load()

            with self.assertRaisesRegex(ProfileError, "selected single-qubit projection"):
                profile.save(selected)

    def test_profile_accepts_pulse_amplitude_at_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            pulses_path = root / "main" / "pulses.json"
            pulses = json.loads(pulses_path.read_text())
            pulses["pulses"]["q1"]["x180_const"]["amplitude"] = 1.0
            pulses_path.write_text(json.dumps(pulses) + "\n", encoding="utf-8")

            Profile("main", root=root).load()

    def test_profile_rejects_pulse_amplitude_above_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            pulses_path = root / "main" / "pulses.json"
            pulses = json.loads(pulses_path.read_text())
            pulses["pulses"]["q1"]["x180_const"]["amplitude"] = 1.01
            pulses_path.write_text(json.dumps(pulses) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "amplitude is too high"):
                Profile("main", root=root).load()

    def test_profile_accepts_optional_gef_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            qubits_path = root / "main" / "qubits.json"
            qubits = json.loads(qubits_path.read_text())
            qubits["qubits"]["q1"]["readout"]["gef_centers"] = [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
            qubits_path.write_text(json.dumps(qubits) + "\n", encoding="utf-8")

            documents = Profile("main", root=root).load()

            self.assertEqual(
                documents["qubits"]["qubits"]["q1"]["readout"]["gef_centers"],
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            )

    def test_profile_accepts_two_state_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            manifest_path = root / "main" / "profile.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["readout_discriminator"] = "nearest_center"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            qubits_path = root / "main" / "qubits.json"
            qubits = json.loads(qubits_path.read_text())
            qubits["qubits"]["q1"]["readout"]["gef_centers"] = [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
            qubits_path.write_text(json.dumps(qubits) + "\n", encoding="utf-8")

            documents = Profile("main", root=root).load()

            self.assertEqual(
                len(documents["qubits"]["qubits"]["q1"]["readout"]["gef_centers"]),
                2,
            )

    def test_profile_rejects_bad_gef_centers_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            qubits_path = root / "main" / "qubits.json"
            qubits = json.loads(qubits_path.read_text())
            qubits["qubits"]["q1"]["readout"]["gef_centers"] = [[0.1, 0.2]]
            qubits_path.write_text(json.dumps(qubits) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "readout.gef_centers"):
                Profile("main", root=root).load()

    def test_profile_rejects_unknown_readout_discriminator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            manifest_path = root / "main" / "profile.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["readout_discriminator"] = "mystery"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "readout_discriminator"):
                Profile("main", root=root).load()

    def test_nearest_center_discriminator_requires_active_qubit_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            manifest_path = root / "main" / "profile.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["readout_discriminator"] = "nearest_center"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "needs readout.gef_centers"):
                Profile("main", root=root).load()

    def test_profile_accepts_optional_readout_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            qubits_path = root / "main" / "qubits.json"
            qubits = json.loads(qubits_path.read_text())
            matrix = [[0.95, 0.05], [0.08, 0.92]]
            qubits["qubits"]["q1"]["readout"]["confusion_matrix"] = matrix
            qubits_path.write_text(json.dumps(qubits) + "\n", encoding="utf-8")

            documents = Profile("main", root=root).load()

            self.assertEqual(
                documents["qubits"]["qubits"]["q1"]["readout"]["confusion_matrix"],
                matrix,
            )

    def test_profile_rejects_bad_readout_confusion_matrix_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_profile(root)
            qubits_path = root / "main" / "qubits.json"
            qubits = json.loads(qubits_path.read_text())
            qubits["qubits"]["q1"]["readout"]["confusion_matrix"] = [[1.0, 0.0]]
            qubits_path.write_text(json.dumps(qubits) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ProfileError, "readout.confusion_matrix"):
                Profile("main", root=root).load()


if __name__ == "__main__":
    unittest.main()
