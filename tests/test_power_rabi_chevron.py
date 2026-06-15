import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.power_rabi_chevron.analysis import process_raw_dataset
from calibration_utils.power_rabi_chevron.parameters import Parameters
from calibration_utils.power_rabi_chevron.plotting import plot_raw_data


REPOSITORY_ROOT = Path(__file__).parent.parent


class PowerRabiChevronTests(unittest.TestCase):
    def test_sequence_sweeps_frequency_and_amplitude(self):
        source = (REPOSITORY_ROOT / "calibrations" / "04d_power_rabi_chevron.py").read_text()

        self.assertIn("with for_(*from_array(df, dfs)):", source)
        self.assertIn("with for_(*from_array(a, amps)):", source)
        self.assertIn("qubit.xy.play(operation, amplitude_scale=a)", source)
        self.assertNotIn("duration=t", source)
        self.assertIn('\"amp_prefactor\": xr.DataArray(', source)
        self.assertIn('\"detuning\": xr.DataArray(', source)
        self.assertIn("qubit.readout_state(state[i])", source)
        self.assertIn("CalibrationSaver().save_xarray(", source)
        self.assertIn("CalibrationSaver().save_figures(", source)
        self.assertIn("use_state_discrimination=node.parameters.use_state_discrimination", source)

    def test_default_amplitude_sweep_is_valid(self):
        parameters = Parameters()
        amps = np.arange(
            parameters.min_amp_factor,
            parameters.max_amp_factor,
            parameters.amp_factor_step,
        )

        self.assertGreater(amps.size, 1)
        self.assertTrue(np.all(np.abs(amps) < 2))

    def test_analysis_adds_frequency_and_actual_amplitude(self):
        ds = self._make_dataset()
        qubit = SimpleNamespace(
            xy=SimpleNamespace(
                RF_frequency=4.1e9,
                operations={"x180": SimpleNamespace(amplitude=0.1)},
            )
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=True, operation="x180"),
            namespace={"qubits": [qubit]},
        )

        processed = process_raw_dataset(ds, node)

        np.testing.assert_allclose(processed.full_freq, [[4.099e9, 4.101e9]])
        np.testing.assert_allclose(processed.full_amp, [[0.05, 0.1]])

    def test_state_plot_uses_amplitude_on_y_axis(self):
        ds = self._make_dataset().assign_coords(
            full_freq=(("qubit", "detuning"), [[4.099e9, 4.101e9]]),
            full_amp=(("qubit", "amp_prefactor"), [[0.05, 0.1]]),
        )
        qubit = SimpleNamespace(name="q7", grid_location="0,0")

        figure = plot_raw_data(ds, [qubit], use_state_discrimination=True)

        self.assertEqual(figure.axes[0].get_ylabel(), "Pulse amplitude [mV]")
        self.assertEqual(figure.axes[0].get_xlabel(), "RF frequency [GHz]")

    def test_iq_plot_shows_i_and_q_quadratures(self):
        ds = xr.Dataset(
            {
                "I": (
                    ("qubit", "detuning", "amp_prefactor"),
                    np.zeros((1, 2, 2)),
                ),
                "Q": (
                    ("qubit", "detuning", "amp_prefactor"),
                    np.ones((1, 2, 2)),
                ),
            },
            coords={
                "qubit": ["q7"],
                "detuning": [-1e6, 1e6],
                "amp_prefactor": [0.5, 1.0],
                "full_freq": (("qubit", "detuning"), [[4.099e9, 4.101e9]]),
                "full_amp": (("qubit", "amp_prefactor"), [[0.05, 0.1]]),
            },
        )
        qubit = SimpleNamespace(name="q7", grid_location="0,0")

        figure = plot_raw_data(ds, [qubit], use_state_discrimination=False)

        data_axes = [axis for axis in figure.axes if axis.get_xlabel() == "RF frequency [GHz]"]
        self.assertEqual(len(data_axes), 2)
        self.assertIn("I [mV]", data_axes[0].get_title())
        self.assertIn("Q [mV]", data_axes[1].get_title())
        self.assertEqual(data_axes[0].get_ylabel(), "Pulse amplitude [mV]")

    @staticmethod
    def _make_dataset():
        return xr.Dataset(
            {
                "state": (
                    ("qubit", "detuning", "amp_prefactor"),
                    np.zeros((1, 2, 2)),
                )
            },
            coords={
                "qubit": ["q7"],
                "detuning": [-1e6, 1e6],
                "amp_prefactor": [0.5, 1.0],
            },
        )


if __name__ == "__main__":
    unittest.main()
