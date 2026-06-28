import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class OrderedCalibration(FakeCalibration):
    def update_state(self):
        super().update_state()
        self.results.setdefault("lifecycle_order", []).append("update_state")

    def propose_profile_update(self, *, apply: bool = True):
        self.results.setdefault("lifecycle_order", []).append("propose_profile_update")
        return True


class FakeMachine:
    def __init__(self):
        self.qmm = Mock()

    def connect(self):
        return self.qmm

    def generate_config(self):
        return {"config": "fake"}


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
            parameters = calibration.namespace["calibration_run_directory"] / "parameters.json"
            self.assertTrue(parameters.is_file())
            self.assertIn('"num_shots": 3', parameters.read_text(encoding="utf-8"))

    def test_run_updates_state_before_profile_proposal(self):
        calibration = OrderedCalibration(
            name="ordered_calibration",
            parameters=SimpleNamespace(simulate=False, load_data_id=None, num_shots=3),
            machine=object(),
            logger=lambda message: None,
            options=CalibrationOptions(
                save_raw_data=False,
                save_figures=False,
                plot_data=False,
            ),
        )

        status = calibration.run()

        self.assertTrue(status.profile_update_proposed)
        self.assertEqual(
            calibration.results["lifecycle_order"],
            ["update_state", "propose_profile_update"],
        )

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

    def test_simulation_shows_figure_when_plotting_is_enabled(self):
        calibration = FakeCalibration(
            name="fake_calibration",
            parameters=SimpleNamespace(
                simulate=True,
                load_data_id=None,
                simulation_duration_ns=1000,
                timeout=100,
                use_waveform_report=False,
            ),
            machine=FakeMachine(),
            logger=lambda message: None,
        )

        with patch(
            "utils.simulation.simulate_and_plot",
            return_value=("samples", "figure", "report"),
        ), patch("calibrations_v2.base.plt.show") as show:
            status = calibration.run()

        self.assertEqual(status.mode, "simulate")
        self.assertEqual(calibration.results["simulation"]["figure"], "figure")
        show.assert_called_once_with()

    def test_simulation_respects_plotting_option(self):
        calibration = FakeCalibration(
            name="fake_calibration",
            parameters=SimpleNamespace(
                simulate=True,
                load_data_id=None,
                simulation_duration_ns=1000,
                timeout=100,
                use_waveform_report=False,
            ),
            machine=FakeMachine(),
            logger=lambda message: None,
            options=CalibrationOptions(plot_data=False),
        )

        with patch(
            "utils.simulation.simulate_and_plot",
            return_value=("samples", "figure", "report"),
        ), patch("calibrations_v2.base.plt.show") as show:
            calibration.run()

        show.assert_not_called()

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
