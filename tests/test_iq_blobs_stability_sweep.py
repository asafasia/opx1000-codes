import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from calibration_io import CalibrationSaver
from sweeps.iq_blobs_stability_sweep import (
    IqBlobsStabilitySweep,
    IqBlobsStabilitySweepParameters,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def time(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeMachine(dict):
    def connect(self):
        self["connected"] = True
        self.qmm = SimpleNamespace(close_all_qms=lambda: self.__setitem__("closed", True))
        return self.qmm


class FakeIqBlobs:
    calls = 0

    def __init__(self, *, parameters, machine, options):
        self.parameters = parameters
        self.machine = machine
        self.options = options
        self.results = {}

    def run(self):
        self.__class__.calls += 1
        point = self.__class__.calls
        self.results["fit_results"] = {
            "q1": {
                "success": True,
                "readout_fidelity": 90.0 + point,
                "readout_fidelity_std": 0.2,
                "average_fidelity": 90.0 + point,
                "average_fidelity_std": 0.2,
                "separation_to_width": 1.5 + 0.1 * point,
                "center_separation": 1e-4,
                "iw_angle": 0.01 * point,
                "ge_threshold": 2e-5,
            }
        }
        return SimpleNamespace(mode="execute")


class InterruptingFakeIqBlobs(FakeIqBlobs):
    def run(self):
        if self.calls >= 1:
            raise KeyboardInterrupt
        return super().run()


class IqBlobsStabilitySweepTests(unittest.TestCase):
    def test_stability_sweep_runs_requested_points_and_saves_summary(self):
        FakeIqBlobs.calls = 0
        clock = FakeClock()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")

            parameters = IqBlobsStabilitySweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                duration_seconds=30 * 60,
                interval_seconds=0,
                max_points=3,
                save_results=True,
                plot_results=True,
            )
            with patch("sweeps.iq_blobs_stability_sweep.IqBlobs", FakeIqBlobs), patch(
                "sweeps.iq_blobs_stability_sweep.plt.show"
            ):
                sweep = IqBlobsStabilitySweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                    machine_factory=lambda **kwargs: FakeMachine(kwargs),
                    time_fn=clock.time,
                    sleep_fn=clock.sleep,
                )
                results = sweep.run()

            self.assertEqual(FakeIqBlobs.calls, 3)
            np.testing.assert_allclose(results["readout_fidelity"], [[91.0, 92.0, 93.0]])
            np.testing.assert_allclose(results["mean_readout_fidelity"], [92.0])
            np.testing.assert_allclose(results["readout_fidelity_stability_span"], [2.0])
            self.assertTrue((sweep.run_directory / "summary.json").is_file())
            self.assertTrue(
                (sweep.run_directory / "figures" / "q1_iq_blobs_stability.png").is_file()
            )
            summary = json.loads((sweep.run_directory / "summary.json").read_text())
            self.assertEqual(summary["completed_points"], 3)
            with np.load(sweep.run_directory / "results.npz") as saved:
                self.assertIn("readout_fidelity_stability_std", saved.files)
                self.assertIn("separation_to_width", saved.files)

    def test_stability_sweep_saves_partial_results_on_interrupt(self):
        InterruptingFakeIqBlobs.calls = 0
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profiles" / "single_qubit"
            profile.mkdir(parents=True)
            (profile / "profile.json").write_text("{}\n", encoding="utf-8")

            parameters = IqBlobsStabilitySweepParameters(
                qubit="q1",
                profile_name="single_qubit",
                duration_seconds=60,
                save_results=True,
                plot_results=False,
            )
            with patch(
                "sweeps.iq_blobs_stability_sweep.IqBlobs",
                InterruptingFakeIqBlobs,
            ):
                sweep = IqBlobsStabilitySweep(
                    parameters,
                    saver=CalibrationSaver(root / "data" / "calibrations", root / "profiles"),
                    machine_factory=lambda **kwargs: FakeMachine(kwargs),
                    time_fn=clock.time,
                    sleep_fn=clock.sleep,
                )
                with self.assertRaises(KeyboardInterrupt):
                    sweep.run()

            self.assertTrue((sweep.run_directory / "summary.json").is_file())
            summary = json.loads((sweep.run_directory / "summary.json").read_text())
            self.assertTrue(summary["interrupted"])
            self.assertEqual(summary["completed_points"], 1)


if __name__ == "__main__":
    unittest.main()
