import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.qubit_spectroscopy.plotting import plot_raw_data_with_fit
from calibration_utils.resonator_spectroscopy.plotting import (
    plot_iq_blobs_for_frequency,
    plot_raw_amplitude,
)
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
                operations={
                    "saturation": SimpleNamespace(amplitude=0.2, length=50000),
                    "x180": SimpleNamespace(amplitude=0.2, length=40),
                },
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
        self.assertIn("pulse amp=20.000 mV, 1.250 MHz", parameter_text.get_text())
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

    def test_qubit_plot_shows_lorentzian_fit_and_measured_max(self):
        detuning = np.linspace(-10e6, 10e6, 41)
        current_frequency = 4.35e9
        center = 1.37e6
        gamma = 1.8e6
        state = 0.1 + 0.8 * gamma**2 / ((detuning - center) ** 2 + gamma**2)
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
                "fit_width": ("qubit", [2 * gamma]),
                "fit_r_squared": ("qubit", [0.999]),
                "fit_offset": ("qubit", [0.1]),
                "fit_amplitude": ("qubit", [0.8]),
                "fit_gamma": ("qubit", [gamma]),
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

        self.assertTrue(any(label.startswith("Lorentzian fit R^2=") for label in labels))
        self.assertTrue(any(label.startswith("Measured max:") for label in labels))

    def test_iq_qubit_plot_shows_lorentzian_fit_on_selected_quadrature(self):
        detuning = np.linspace(-10e6, 10e6, 41)
        current_frequency = 4.35e9
        center = 1.37e6
        gamma = 1.8e6
        ds = xr.Dataset(
            {
                "I": (("qubit", "detuning"), np.zeros((1, detuning.size))),
                "Q": (
                    ("qubit", "detuning"),
                    (0.1 + 0.8 * gamma**2 / ((detuning - center) ** 2 + gamma**2))[
                        np.newaxis,
                        :,
                    ],
                ),
            },
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
                "measured_max_position": ("qubit", [detuning[np.argmax(ds.Q.values[0])]]),
                "fit_position": ("qubit", [center]),
                "fit_width": ("qubit", [2 * gamma]),
                "fit_r_squared": ("qubit", [0.999]),
                "fit_offset": ("qubit", [0.1]),
                "fit_amplitude": ("qubit", [0.8]),
                "fit_gamma": ("qubit", [gamma]),
                "selected_quadrature": ("qubit", ["Q"]),
            },
            coords={"qubit": ["q1"]},
        )
        qubit = SimpleNamespace(
            name="q1",
            f_01=current_frequency,
            xy=SimpleNamespace(RF_frequency=current_frequency),
        )

        fig = plot_raw_data_with_fit(ds, [qubit], fits, use_state_discrimination=False)
        labels_by_title = {
            axis.get_title(): axis.get_legend_handles_labels()[1] for axis in fig.axes
        }

        self.assertFalse(
            any(
                label.startswith("Lorentzian fit R^2=")
                for label in labels_by_title["q1: I [mV]"]
            )
        )
        self.assertTrue(
            any(
                label.startswith("Lorentzian fit R^2=")
                for label in labels_by_title["q1: Q [mV]"]
            )
        )

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

    def test_ef_plot_does_not_add_rotated_subplot(self):
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
        fits = xr.Dataset(
            {
                "res_freq": ("qubit", [4.201e9]),
                "I_rot": ("qubit", [0.0]),
            },
            coords={"qubit": ["q1"]},
        )
        qubit = SimpleNamespace(
            name="q1",
            f_01=4.35e9,
            f_12=4.2e9,
            anharmonicity=150e6,
            xy=SimpleNamespace(RF_frequency=4.35e9),
        )

        fig = plot_raw_data_with_fit(ds, [qubit], fits, transition="ef")

        titles = [axis.get_title() for axis in fig.axes]
        self.assertIn("q1: I [mV]", titles)
        self.assertIn("q1: Q [mV]", titles)
        self.assertNotIn("q1: Rotated I [mV]", titles)

    def test_ge_plot_does_not_add_rotated_subplot(self):
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
        fits = xr.Dataset(
            {
                "res_freq": ("qubit", [4.351e9]),
                "I_rot": ("qubit", [0.0]),
            },
            coords={"qubit": ["q1"]},
        )
        qubit = SimpleNamespace(
            name="q1",
            f_01=4.35e9,
            xy=SimpleNamespace(RF_frequency=4.35e9),
        )

        fig = plot_raw_data_with_fit(ds, [qubit], fits)

        titles = [axis.get_title() for axis in fig.axes]
        self.assertIn("q1: I [mV]", titles)
        self.assertIn("q1: Q [mV]", titles)
        self.assertNotIn("q1: Rotated I [mV]", titles)

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
                "detuning": np.linspace(-5e6, 5e6, 11),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        qubit = SimpleNamespace(
            name="q1",
            grid_location="0,0",
            resonator=SimpleNamespace(
                RF_frequency=7.47e9,
                operations={"readout": SimpleNamespace(length=1200, amplitude=0.045)},
            ),
            xy=SimpleNamespace(
                operations={
                    "saturation": SimpleNamespace(amplitude=0.1, length=30000),
                    "x180": SimpleNamespace(amplitude=0.2, length=40),
                },
            ),
        )

        fig = plot_raw_amplitude(
            ds,
            [qubit],
            qubit_operation="saturation",
            saturation_amplitude_factor=0.5,
            saturation_lead_time_in_ns=10000,
        )
        labels = [label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]]

        self.assertTrue(any(label.startswith("Current resonance:") for label in labels))
        self.assertTrue(any(label.startswith("New resonance") for label in labels))
        parameter_text = next(
            text for text in fig.texts if text.get_gid() == "resonator_spectroscopy_parameters"
        )
        self.assertIn("frequency span=10 MHz", parameter_text.get_text())
        self.assertIn("readout length=1200 ns", parameter_text.get_text())
        self.assertIn("readout amp=0.045 V", parameter_text.get_text())
        self.assertIn("driven operation=saturation", parameter_text.get_text())
        self.assertIn("amplitude factor=0.5", parameter_text.get_text())
        self.assertIn("lead time=10000 ns", parameter_text.get_text())
        self.assertIn("drive amp=50.000 mV, 3.125 MHz", parameter_text.get_text())
        for axis in fig.axes:
            self.assertEqual(len(axis.child_axes), 1)
            self.assertEqual(
                axis.child_axes[0].get_xlabel(),
                "Detuning from current resonance [MHz]",
            )

    def test_resonator_frequency_iq_blobs_uses_nearest_sweep_index(self):
        frequencies = np.asarray([6.873e9, 6.874e9, 6.875e9, 6.876e9])
        values = np.arange(8, dtype=float).reshape(1, 2, 4) * 1e-3
        ds = xr.Dataset(
            {
                "Ig": (("qubit", "n_runs", "detuning"), values),
                "Qg": (("qubit", "n_runs", "detuning"), values + 1e-3),
                "Im": (("qubit", "n_runs", "detuning"), values + 2e-3),
                "Qm": (("qubit", "n_runs", "detuning"), values + 3e-3),
            },
            coords={
                "qubit": ["q3"],
                "n_runs": [0, 1],
                "detuning": np.asarray([-2e6, -1e6, 0.0, 1e6]),
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        qubit = SimpleNamespace(
            name="q3",
            grid_location="0,0",
            resonator=SimpleNamespace(RF_frequency=6.875e9),
        )

        fig = plot_iq_blobs_for_frequency(ds, [qubit], 6.875)

        self.assertIn("index 2", fig.axes[0].get_title())
        parameter_text = next(
            text
            for text in fig.texts
            if text.get_gid() == "resonator_spectroscopy_frequency_iq_parameters"
        )
        self.assertIn("q3: index=2", parameter_text.get_text())

    def test_resonator_second_subplot_shows_readout_fidelity(self):
        frequencies = np.asarray([7.469e9, 7.470e9, 7.471e9])
        ds = xr.Dataset(
            {
                "ground_IQ_abs": (
                    ("qubit", "detuning"),
                    [[1.0e-3, 1.1e-3, 1.0e-3]],
                ),
                "mixed_IQ_abs": (
                    ("qubit", "detuning"),
                    [[1.2e-3, 1.3e-3, 1.2e-3]],
                ),
                "IQ_separation": (
                    ("qubit", "detuning"),
                    [[1.0, 4.0, 2.0]],
                ),
                "readout_fidelity": (
                    ("qubit", "detuning"),
                    [[75.0, 96.5, 82.0]],
                ),
            },
            coords={
                "qubit": ["q1"],
                "detuning": [-1e6, 0.0, 1e6],
                "full_freq": (("qubit", "detuning"), frequencies[None, :]),
            },
        )
        qubit = SimpleNamespace(
            name="q1",
            grid_location="0,0",
            resonator=SimpleNamespace(RF_frequency=7.470e9),
        )

        fig = plot_raw_amplitude(ds, [qubit])

        fidelity_ax = next(
            axis
            for axis in fig.axes
            if axis.get_ylabel() == "Optimal discrimination fidelity [%]"
        )
        fidelity_line = next(
            line for line in fidelity_ax.lines if line.get_label() == "Readout fidelity"
        )
        np.testing.assert_allclose(fidelity_line.get_ydata(), [75.0, 96.5, 82.0])
        self.assertEqual(tuple(fidelity_ax.get_ylim()), (75.0, 100.0))


if __name__ == "__main__":
    unittest.main()
