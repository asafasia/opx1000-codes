import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from profiles import Profile, clear_active_profile, set_active_profile
from calibration_io import CalibrationSaver
from utils.plotting_settings import CALIBRATION_TIMESTAMP_GID, add_calibration_timestamp


class FakeDataArray:
    def __init__(self, values):
        self.values = values


class FakeDataset:
    def __init__(self):
        self.coords = {"qubit": FakeDataArray(np.array(["q1", "q2"], dtype=object))}
        self.data_vars = {"I": FakeDataArray(np.array([[0.1, 0.2], [0.3, 0.4]]))}


class CalibrationSaverTests(unittest.TestCase):
    def tearDown(self):
        clear_active_profile()

    def test_save_creates_dated_run_with_arrays_and_profile_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text('{"name": "main"}\n', encoding="utf-8")
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")
            timestamp = datetime(2026, 6, 10, 14, 5, 6, 123456, tzinfo=timezone.utc)

            run_directory = saver.save(
                "qubit_spectroscopy",
                sweep={"detuning": [1, 2, 3]},
                results={"I": [0.1, 0.2, 0.3], "Q": [0.4, 0.5, 0.6]},
                profile_name="main",
                now=timestamp,
            )

            self.assertEqual(
                run_directory,
                root / "data" / "calibrations" / "2026-06-10" / "qubit_spectroscopy" / "14-05-06-123456",
            )
            np.testing.assert_array_equal(np.load(run_directory / "sweep.npz")["detuning"], [1, 2, 3])
            np.testing.assert_allclose(np.load(run_directory / "results.npz")["I"], [0.1, 0.2, 0.3])
            self.assertEqual(
                (run_directory / "profile" / "profile.json").read_text(encoding="utf-8"),
                '{"name": "main"}\n',
            )
            metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["profile_name"], "main")
            self.assertEqual(metadata["results"]["Q"]["shape"], [3])

    def test_save_writes_parameters_with_arrays_and_profile_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text('{"name": "main"}\n', encoding="utf-8")
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")

            run_directory = saver.save(
                "iq_blobs",
                sweep={"shot": [0, 1]},
                results={"I": [0.1, 0.2]},
                profile_name="main",
                parameters={"num_shots": np.int64(2), "reset_type": "active"},
            )

            self.assertTrue((run_directory / "sweep.npz").is_file())
            self.assertTrue((run_directory / "results.npz").is_file())
            self.assertTrue((run_directory / "profile" / "profile.json").is_file())
            saved_parameters = json.loads((run_directory / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_parameters, {"num_shots": 2, "reset_type": "active"})
            metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["parameters"], "parameters.json")

    def test_save_xarray_handles_object_typed_string_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")

            run_directory = saver.save_xarray("resonator_spectroscopy", FakeDataset(), profile_name="main")

            np.testing.assert_array_equal(
                np.load(run_directory / "sweep.npz", allow_pickle=False)["qubit"],
                ["q1", "q2"],
            )

    def test_save_defaults_to_active_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text(
                '{"name": "single_qubit"}\n',
                encoding="utf-8",
            )
            set_active_profile(Profile("single_qubit", qubit="q3", root=root / "profiles"))
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")

            run_directory = saver.save("iq_blobs", [1], [2])

            metadata = json.loads((run_directory / "metadata.json").read_text())
            self.assertEqual(metadata["profile_name"], "single_qubit")
            self.assertEqual(
                (run_directory / "profile" / "profile.json").read_text(encoding="utf-8"),
                '{"name": "single_qubit"}\n',
            )

    def test_save_accepts_profile_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text(
                '{"name": "single_qubit"}\n',
                encoding="utf-8",
            )
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")

            run_directory = saver.save(
                "iq_blobs",
                [1],
                [2],
                profile_name=Profile("single_qubit", qubit="q3", root=root / "profiles"),
            )

            metadata = json.loads((run_directory / "metadata.json").read_text())
            self.assertEqual(metadata["profile_name"], "single_qubit")

    def test_save_rejects_path_traversal_in_names(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = CalibrationSaver(Path(directory) / "data" / "calibrations", Path(directory) / "profiles")

            with self.assertRaises(ValueError):
                saver.save("../outside", [1], [2])

    def test_save_figures_adds_pngs_to_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            run_directory.mkdir()
            (run_directory / "metadata.json").write_text(
                '{"timestamp": "2026-06-14T09:30:00+03:00"}\n',
                encoding="utf-8",
            )
            figure = Figure()
            figures_directory = CalibrationSaver().save_figures(
                run_directory,
                {"iq_blobs": figure},
            )

            self.assertEqual(figures_directory, run_directory / "figures")
            self.assertTrue((figures_directory / "iq_blobs.png").is_file())
            timestamps = [text for text in figure.texts if text.get_gid() == CALIBRATION_TIMESTAMP_GID]
            self.assertEqual(len(timestamps), 1)
            self.assertIn("2026-06-14T09:30:00+03:00", timestamps[0].get_text())

    def test_save_figures_does_not_duplicate_existing_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            run_directory.mkdir()
            figure = Figure()
            add_calibration_timestamp(figure, "already stamped")

            CalibrationSaver().save_figures(run_directory, {"figure": figure})

            timestamps = [text for text in figure.texts if text.get_gid() == CALIBRATION_TIMESTAMP_GID]
            self.assertEqual(len(timestamps), 1)


if __name__ == "__main__":
    unittest.main()
