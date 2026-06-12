import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.qubit_spectroscopy.plotting import plot_raw_data_with_fit
from calibration_utils.resonator_spectroscopy.plotting import plot_raw_amplitude


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
        qubit = SimpleNamespace(name="q1", xy=SimpleNamespace(RF_frequency=4.3496e9))

        fig = plot_raw_data_with_fit(ds, [qubit], fits)
        labels = [label for axis in fig.axes for label in axis.get_legend_handles_labels()[1]]

        self.assertTrue(any(label.startswith("Current resonance:") for label in labels))
        self.assertTrue(any(label.startswith("New resonance:") for label in labels))
        self.assertEqual(len(fig.axes[0].child_axes), 1)
        self.assertEqual(
            fig.axes[0].child_axes[0].get_xlabel(),
            "Detuning from current resonance [MHz]",
        )

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
