import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_io import CalibrationSaver
from calibrations_v2 import BaseCalibration, CalibrationOptions
from profiles import Profile, clear_active_profile, set_active_profile


class FakeCalibration(BaseCalibration[SimpleNamespace, object]):
    def create_qua_program(self):
        self.namespace["sweep_axes"] = {"x": xr.DataArray(np.array([1, 2, 3]))}
        return "program"

    def execute_qua_program(self):
        self.results["ds_raw"] = xr.Dataset(
            data_vars={"signal": ("x", np.array([0.1, 0.2, 0.3]))},
            coords={"x": np.array([1, 2, 3])},
        )

    def analyse_data(self):
        self.results["analysed"] = True
        self.outcomes = {"q1": "successful"}

    def plot_data(self):
        self.results["plotted"] = True

    def update_state(self):
        self.results["updated"] = True


class CalibrationsV2BaseTests(unittest.TestCase):
    def tearDown(self):
        clear_active_profile()

    def test_run_executes_lifecycle_and_saves_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            set_active_profile(Profile("main", root=root / "profiles"))

            calibration = FakeCalibration(
                name="fake_calibration",
                parameters=SimpleNamespace(simulate=False, load_data_id=None, num_shots=3),
                machine=object(),
                saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                profile_name="main",
                logger=lambda message: None,
            )

            status = calibration.run()

            self.assertEqual(status.mode, "execute")
            self.assertTrue(status.raw_data_saved)
            self.assertEqual(status.outcomes, {"q1": "successful"})
            self.assertTrue(calibration.results["analysed"])
            self.assertTrue((calibration.namespace["calibration_run_directory"] / "results.npz").is_file())

    def test_options_can_skip_saving_plotting_and_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            set_active_profile(Profile("main", root=root / "profiles"))

            calibration = FakeCalibration(
                name="fake_calibration",
                parameters=SimpleNamespace(simulate=False, load_data_id=None, num_shots=3),
                machine=object(),
                saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                profile_name="main",
                logger=lambda message: None,
                options=CalibrationOptions(
                    save_raw_data=False,
                    save_figures=False,
                    plot_data=False,
                    update_state=False,
                    propose_profile_update=False,
                ),
            )

            status = calibration.run()

            self.assertFalse(status.raw_data_saved)
            self.assertNotIn("calibration_run_directory", calibration.namespace)
            self.assertTrue(calibration.results["analysed"])
            self.assertNotIn("plotted", calibration.results)
            self.assertNotIn("updated", calibration.results)

    def test_load_saved_run_reconstructs_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "main"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")
            saver = CalibrationSaver(root / "data" / "calibrations", root / "profiles")
            dataset = xr.Dataset(
                data_vars={"signal": ("x", np.array([4.0, 5.0]))},
                coords={"x": np.array([10, 20])},
            )
            run_directory = saver.save_xarray("fake_calibration", dataset, profile_name="main")

            calibration = FakeCalibration(
                name="fake_calibration",
                parameters=SimpleNamespace(simulate=False, load_data_id=None),
                machine=object(),
                saver=saver,
                profile_name="main",
                logger=lambda message: None,
            )

            loaded = calibration.load_saved_run(run_directory)

            np.testing.assert_array_equal(loaded.coords["x"].values, [10, 20])
            np.testing.assert_allclose(loaded.data_vars["signal"].values, [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
