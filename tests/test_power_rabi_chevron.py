import unittest
import importlib
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.power_rabi_chevron.analysis import process_raw_dataset, pulse_rabi_calibration
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
        self.assertIn("self.readout_state(qubit, state[i])", source)
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
                operations={"x180": SimpleNamespace(amplitude=0.1, length=40, calculate_waveform=lambda: 0.1)},
            )
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=True, operation="x180"),
            namespace={"qubits": [qubit]},
        )

        processed = process_raw_dataset(ds, node)

        np.testing.assert_allclose(processed.full_freq, [[4.099e9, 4.101e9]])
        np.testing.assert_allclose(processed.full_amp, [[0.05, 0.1]])
        np.testing.assert_allclose(processed.rabi_frequency_hz, [[6.25e6, 12.5e6]])

    def test_state_plot_uses_amplitude_on_y_axis(self):
        ds = self._make_dataset().assign_coords(
            full_freq=(("qubit", "detuning"), [[4.099e9, 4.101e9]]),
            full_amp=(("qubit", "amp_prefactor"), [[0.05, 0.1]]),
            rabi_frequency_hz=(("qubit", "amp_prefactor"), [[6.25e6, 12.5e6]]),
        )
        qubit = SimpleNamespace(name="q7", grid_location="0,0")

        figure = plot_raw_data(ds, [qubit], use_state_discrimination=True)

        self.assertEqual(figure.axes[0].get_ylabel(), "Rabi frequency [MHz]")
        self.assertEqual(figure.axes[0].get_xlabel(), "RF frequency [GHz]")
        self.assertTrue(
            any(axis.get_xlabel() == "Detuning [MHz]" for axis in figure.axes)
        )
        self.assertTrue(self._has_secondary_amplitude_yaxis(figure.axes[0]))
        self.assertEqual(figure._suptitle.get_text(), "Power Rabi chevron: measured state")

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
                "rabi_frequency_hz": (("qubit", "amp_prefactor"), [[6.25e6, 12.5e6]]),
            },
        )
        qubit = SimpleNamespace(name="q7", grid_location="0,0")

        figure = plot_raw_data(ds, [qubit], use_state_discrimination=False)

        data_axes = [axis for axis in figure.axes if axis.get_xlabel() == "RF frequency [GHz]"]
        self.assertEqual(len(data_axes), 2)
        self.assertIn("I [mV]", data_axes[0].get_title())
        self.assertIn("Q [mV]", data_axes[1].get_title())
        self.assertEqual(data_axes[0].get_ylabel(), "Rabi frequency [MHz]")
        self.assertTrue(self._has_secondary_amplitude_yaxis(data_axes[0]))
        self.assertEqual(figure._suptitle.get_text(), "Power Rabi chevron: I and Q quadratures")
        self.assertLess(data_axes[1].get_position().y1, data_axes[0].get_position().y0)

    def test_cosine_reference_uses_sampled_area_not_duration(self):
        from quam.components.pulses import SquarePulse
        # A 40-sample cosine has area 19.5 * amplitude ns, not 40 * amplitude ns.
        reference_samples = 0.2 * (1 - np.cos(np.linspace(0, 2 * np.pi, 40))) / 2
        reference = SimpleNamespace(amplitude=0.2, length=40, axis_angle=np.pi / 2,
                                    calculate_waveform=lambda: 1j * reference_samples)
        operation = SquarePulse(amplitude=0.1, length=200)
        qubit = SimpleNamespace(xy=SimpleNamespace(operations={"x180": reference, "saturation": operation}))
        gain, area, applicable, status = pulse_rabi_calibration(qubit, "saturation")
        self.assertAlmostEqual(area / 1e-9, 3.9)
        self.assertAlmostEqual(gain * 0.1 / 1e6, 12.8205128205)
        self.assertTrue(applicable)

    def test_missing_or_detuned_reference_disables_physical_theory(self):
        pulse = SimpleNamespace(amplitude=0.1, length=40, detuning=1e6,
                                calculate_waveform=lambda: 0.1)
        qubit = SimpleNamespace(xy=SimpleNamespace(operations={"x180": pulse}))
        gain, _, applicable, status = pulse_rabi_calibration(qubit, "x180")
        self.assertTrue(np.isnan(gain))
        self.assertFalse(applicable)
        self.assertIn("detuning", status)

    def test_analysis_lifecycle_saves_reviewable_peak_table_and_can_disable_fit(self):
        from tests.test_power_rabi_stark import make_spectrum
        from calibrations.core import CalibrationOptions
        from quam.components.pulses import SquarePulse
        cls = importlib.import_module("calibrations.04d_power_rabi_chevron").PowerRabiChevron
        data, _, _ = make_spectrum()
        parameters = Parameters(use_state_discrimination=True, operation="saturation")
        qubit = SimpleNamespace(name="q1", anharmonicity=200e6, xy=SimpleNamespace(
            RF_frequency=5e9, operations={
                "x180": SquarePulse(amplitude=0.125, length=40),
                "saturation": SquarePulse(amplitude=0.2, length=1000),
            }))
        calibration = cls(parameters=parameters, machine=SimpleNamespace(), logger=lambda message: None,
                          options=CalibrationOptions(save_raw_data=False, save_figures=False,
                                                     plot_data=False, update_state=False,
                                                     propose_profile_update=False, apply_profile_update=False))
        calibration.namespace["qubits"] = [qubit]
        calibration.results["ds_raw"] = data[["state"]].drop_vars(
            [name for name in data.coords if name not in ("qubit", "detuning", "amp_prefactor")])
        calibration.analyse_data()
        self.assertTrue(calibration.results["fit_results"]["q1"]["success"])
        self.assertAlmostEqual(calibration.results["fit_results"]["q1"]["theory_coefficient"], 0.5, delta=0.02)
        with patch("matplotlib.pyplot.show"):
            calibration.plot_data()
        self.assertEqual(set(calibration.results["figures"]), {
            "power_rabi_chevron_q1", "ac_stark_shift_q1",
        })
        for figure in calibration.results["figures"].values():
            figure.canvas.draw()
            matplotlib.pyplot.close(figure)
        with tempfile.TemporaryDirectory() as directory:
            calibration.namespace["calibration_run_directory"] = directory
            self.assertTrue(calibration.save_analysis_result())
            import pandas as pd
            table = pd.read_csv(Path(directory) / "stark_peaks.csv")
            self.assertEqual(len(table), data.sizes["amp_prefactor"])
            self.assertIn("peak_status", table.columns)
            self.assertIn("gaussian_initial_center_hz", table.columns)
            self.assertIn("gaussian_sigma_hz", table.columns)
            self.assertIn("gaussian_height_snr", table.columns)
            report = json.loads((Path(directory) / "analysis_result.json").read_text())
            self.assertEqual(report["fit_results"]["q1"]["quadratic_consistency"], "consistent")
        calibration.parameters.fit_stark_shift = False
        calibration.analyse_data()
        self.assertNotIn("ds_fit", calibration.results)
        self.assertEqual(calibration.results["fit_results"], {})
        calibration.namespace.pop("calibration_run_directory")
        with patch("matplotlib.pyplot.show"):
            calibration.plot_data()
        self.assertEqual(set(calibration.results["figures"]), {"power_rabi_chevron_q1"})
        for figure in calibration.results["figures"].values():
            figure.canvas.draw()
            matplotlib.pyplot.close(figure)

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

    @staticmethod
    def _has_secondary_amplitude_yaxis(axis):
        return any(
            type(child).__name__ == "SecondaryAxis"
            and child.get_ylabel() == "Pulse amplitude [mV]"
            for child in axis.get_children()
        )


if __name__ == "__main__":
    unittest.main()
