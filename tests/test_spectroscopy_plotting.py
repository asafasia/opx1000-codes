import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.qubit_spectroscopy.plotting import plot_raw_data_with_fit
from calibration_utils.resonator_spectroscopy.plotting import plot_raw_amplitude
from utils.plotting_settings import CALIBRATION_TIMESTAMP_GID


class SpectroscopyPlottingTests(unittest.TestCase):
    def test_qubit_plot_labels_current_and_new_resonances(self):
        frequencies = np.linspace(4.34e9, 4.36e9, 11)
        ds = xr.Dataset(
            {
                "I": (("qubit", "detuning"), np.zeros((1, 11))),
                "Q": (("qubit", "detuning"), np.zeros((1, 11))),
            },
            coords={
                "qubit": ["q1"],
                "detuning": np.arange(11),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        fits = xr.Dataset({"res_freq": ("qubit", [4.351e9])}, coords={"qubit": ["q1"]})
        qubit = SimpleNamespace(
            name="q1",
            f_01=4.3496e9,
            xy=SimpleNamespace(
                RF_frequency=4.3496e9,
                operations={"saturation": SimpleNamespace(amplitude=0.2, length=50000)},
            ),
        )

        fig = plot_raw_data_with_fit(
            ds,
            [qubit],
            fits,
            operation="saturation",
            operation_amplitude_factor=0.1,
            operation_len_in_ns=1000,
        )
        labels = [label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]]

        self.assertTrue(any(label.startswith("Current drive f01:") for label in labels))
        self.assertTrue(any(label.startswith("Fitted new f01:") for label in labels))
        self.assertEqual(len(fig.axes[0].child_axes), 1)
        self.assertEqual(
            fig.axes[0].child_axes[0].get_xlabel(),
            "Detuning from current resonance [MHz]",
        )
        parameter_text = next(text for text in fig.texts if text.get_gid() == "spectroscopy_parameters")
        self.assertIn("operation=saturation", parameter_text.get_text())
        self.assertIn("pulse length=1000 ns", parameter_text.get_text())
        self.assertIn("pulse amp=20.000 mV", parameter_text.get_text())
        self.assertIn("current drive f01=4.349600 GHz", parameter_text.get_text())
        self.assertIn("fitted/new f01=4.351000 GHz", parameter_text.get_text())
        self.assertEqual(
            len([text for text in fig.texts if text.get_gid() == CALIBRATION_TIMESTAMP_GID]),
            1,
        )

    def test_qubit_plot_shows_only_state_when_discrimination_is_enabled(self):
        frequencies = np.linspace(4.34e9, 4.36e9, 11)
        ds = xr.Dataset(
            {
                "state": (("qubit", "detuning"), np.linspace(0, 1, 11)[None, :]),
                "I": (("qubit", "detuning"), np.zeros((1, 11))),
                "Q": (("qubit", "detuning"), np.zeros((1, 11))),
            },
            coords={
                "qubit": ["q1"],
                "detuning": np.arange(11),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        fits = xr.Dataset({"res_freq": ("qubit", [4.351e9])}, coords={"qubit": ["q1"]})
        qubit = SimpleNamespace(name="q1", f_01=4.3496e9, xy=SimpleNamespace(RF_frequency=4.3496e9))

        fig = plot_raw_data_with_fit(ds, [qubit], fits, use_state_discrimination=True)

        self.assertEqual(fig.axes[0].get_title(), "q1: Measured state")
        self.assertEqual(fig.axes[0].get_ylabel(), "Measured state")
        self.assertNotIn("q1: I [mV]", [axis.get_title() for axis in fig.axes])
        self.assertNotIn("q1: Q [mV]", [axis.get_title() for axis in fig.axes])

    def test_qubit_plot_shows_gaussian_fit_and_measured_max(self):
        detuning = np.linspace(-10e6, 10e6, 41)
        current_frequency = 4.35e9
        center = 1.37e6
        sigma = 1.8e6
        state = 0.1 + 0.8 * np.exp(-0.5 * ((detuning - center) / sigma) ** 2)
        ds = xr.Dataset(
            {"state": (("qubit", "detuning"), state[np.newaxis, :])},
            coords={
                "qubit": ["q1"],
                "detuning": detuning,
                "full_freq": (
                    ("qubit", "detuning"),
                    (current_frequency + detuning)[None, :],
                ),
            },
        )
        fits = xr.Dataset(
            {
                "res_freq": ("qubit", [current_frequency + center]),
                "measured_max_position": ("qubit", [detuning[np.argmax(state)]]),
                "fit_position": ("qubit", [center]),
                "fit_width": ("qubit", [2 * np.sqrt(2 * np.log(2)) * sigma]),
                "fit_r_squared": ("qubit", [0.999]),
                "fit_offset": ("qubit", [0.1]),
                "fit_amplitude": ("qubit", [0.8]),
                "fit_sigma": ("qubit", [sigma]),
            },
            coords={"qubit": ["q1"]},
        )
        qubit = SimpleNamespace(
            name="q1",
            f_01=current_frequency,
            xy=SimpleNamespace(RF_frequency=current_frequency),
        )

        fig = plot_raw_data_with_fit(ds, [qubit], fits, use_state_discrimination=True)
        labels = [
            label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]
        ]

        self.assertTrue(any(label.startswith("Gaussian fit R^2=") for label in labels))
        self.assertTrue(any(label.startswith("Measured max:") for label in labels))

    def test_ef_plot_marks_current_ge_and_ef_without_expanding_sweep_limits(self):
        frequencies = np.linspace(4.19e9, 4.21e9, 11)
        ds = xr.Dataset(
            {
                "I": (("qubit", "detuning"), np.zeros((1, 11))),
                "Q": (("qubit", "detuning"), np.zeros((1, 11))),
            },
            coords={
                "qubit": ["q1"],
                "detuning": np.arange(11),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        fits = xr.Dataset({"res_freq": ("qubit", [4.201e9])}, coords={"qubit": ["q1"]})
        qubit = SimpleNamespace(
            name="q1",
            f_01=4.35e9,
            f_12=4.2e9,
            anharmonicity=150e6,
            xy=SimpleNamespace(RF_frequency=4.35e9),
        )

        fig = plot_raw_data_with_fit(ds, [qubit], fits, transition="ef")
        labels = [label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]]

        self.assertTrue(any(label.startswith("Current drive f01:") for label in labels))
        self.assertTrue(any(label.startswith("Current ef:") for label in labels))
        self.assertTrue(any(label.startswith("Fitted new ef:") for label in labels))
        self.assertEqual(fig.axes[0].get_xlim(), (4.19, 4.21))

    def test_resonator_plot_labels_current_and_new_resonances(self):
        frequencies = np.linspace(7.46e9, 7.48e9, 11)
        separation = np.zeros((1, 11))
        separation[0, 7] = 1e-3
        ds = xr.Dataset(
            {
                "ground_IQ_abs": (("qubit", "detuning"), np.ones((1, 11)) * 1e-3),
                "mixed_IQ_abs": (("qubit", "detuning"), np.ones((1, 11)) * 1.1e-3),
                "IQ_separation": (("qubit", "detuning"), separation),
            },
            coords={
                "qubit": ["q1"],
                "detuning": np.arange(11),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        qubit = SimpleNamespace(
            name="q1",
            grid_location="0,0",
            resonator=SimpleNamespace(RF_frequency=7.47e9),
        )

        fig = plot_raw_amplitude(ds, [qubit])
        labels = [label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]]

        self.assertTrue(any(label.startswith("Current resonance:") for label in labels))
        self.assertTrue(any(label.startswith("New resonance") for label in labels))
        for axis in fig.axes:
            self.assertEqual(len(axis.child_axes), 1)
            self.assertEqual(
                axis.child_axes[0].get_xlabel(),
                "Detuning from current resonance [MHz]",
            )


if __name__ == "__main__":
    unittest.main()
