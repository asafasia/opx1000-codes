import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.fine_rabi.analysis import analyze_fine_rabi, fit_fourier_branches, process_raw_dataset
from calibration_utils.fine_rabi.parameters import (
    Parameters,
    operation_for_rotation,
    pulses_per_repetition_group,
)
from calibration_utils.fine_rabi.plotting import plot_fine_rabi


REPOSITORY_ROOT = Path(__file__).parent.parent


class FineRabiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            REPOSITORY_ROOT / "calibrations" / "04e_fine_rabi_calibration.py"
        ).read_text()

    def test_rotation_type_maps_to_complete_gate_groups(self):
        self.assertEqual(operation_for_rotation("PI"), "x180")
        self.assertEqual(operation_for_rotation("PI_HALF"), "x90")
        self.assertEqual(pulses_per_repetition_group("PI"), 2)
        self.assertEqual(pulses_per_repetition_group("PI_HALF"), 4)

    def test_sequence_sweeps_amplitude_and_repetition_groups(self):
        self.assertIn("with for_(*from_array(group_count, repetition_groups)):", self.source)
        self.assertIn("with for_each_(a, amps.tolist()):", self.source)
        self.assertNotIn("from_array(a, amps)", self.source)
        self.assertIn("group_index < group_count", self.source)
        self.assertIn("pulse_index < pulses_per_group", self.source)
        self.assertIn("qubit.xy.play(operation, amplitude_scale=a)", self.source)

    def test_sequence_supports_state_and_iq_readout(self):
        self.assertIn("if node.parameters.use_state_discrimination:", self.source)
        self.assertIn("qubit.readout_state(state[i])", self.source)
        self.assertIn('save(f"state{i + 1}")', self.source)
        self.assertIn('save(f"I{i + 1}")', self.source)
        self.assertIn('save(f"Q{i + 1}")', self.source)

    def test_profile_update_scales_x180_amplitude_from_optimum_factor(self):
        self.assertIn("def propose_profile_update", self.source)
        self.assertIn('qubit_profile["operations"]["x180"]', self.source)
        self.assertIn('result["optimal_amp_prefactor"]', self.source)
        self.assertIn("current_amplitude * opt_amp_factor", self.source)
        self.assertIn("pulses.json.pulses.{q.name}.{pulse_name}.amplitude", self.source)
        self.assertIn("ProfileUpdater().stage", self.source)

    def test_parameters_include_amplitude_endpoint(self):
        params = Parameters(min_amp_factor=0.8, max_amp_factor=0.82, amp_factor_step=0.01)

        np.testing.assert_allclose(params.get_amp_factors(), [0.8, 0.81, 0.82])
        self.assertEqual(params.fourier_oversampling, 8)

    def test_parameters_can_cluster_amplitude_points_around_current_amp(self):
        uniform = Parameters(min_amp_factor=0.8, max_amp_factor=1.2, amp_factor_step=0.01)
        dense = Parameters(
            min_amp_factor=0.8,
            max_amp_factor=1.2,
            amp_factor_step=0.01,
            amp_factor_spacing="center_dense",
            amp_factor_density_power=2.0,
        )

        uniform_amps = uniform.get_amp_factors()
        dense_amps = dense.get_amp_factors()

        self.assertEqual(len(dense_amps), len(uniform_amps))
        self.assertAlmostEqual(dense_amps[0], 0.8)
        self.assertAlmostEqual(dense_amps[-1], 1.2)
        self.assertIn(1.0, dense_amps)
        self.assertGreater(
            np.count_nonzero(np.abs(dense_amps - 1.0) <= 0.05),
            np.count_nonzero(np.abs(uniform_amps - 1.0) <= 0.05),
        )

    def test_state_processing_keeps_measured_state(self):
        ds = xr.Dataset(
            {"state": (("qubit", "repetition_group_count", "amp_prefactor"), [[[0.1, 0.8]]])},
            coords={
                "qubit": ["q1"],
                "repetition_group_count": [0],
                "amp_prefactor": [0.9, 1.0],
            },
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                fourier_oversampling=8,
            )
        )

        processed = process_raw_dataset(ds, node)

        self.assertIn("state", processed)
        self.assertNotIn("ground_population", processed)
        np.testing.assert_allclose(processed.state, ds.state)

    def test_plot_has_scan_and_fourier_subplots(self):
        repetition_counts = np.arange(81)
        amp_prefactors = np.linspace(0.8, 1.2, 41)
        values = _make_v_ridge_state(repetition_counts, amp_prefactors, optimum=1.0, slope=1.0)
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "repetition_group_count", "amp_prefactor"),
                    values[None, :, :],
                )
            },
            coords={
                "qubit": ["q1"],
                "repetition_group_count": repetition_counts,
                "amp_prefactor": amp_prefactors,
            },
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                fourier_oversampling=8,
            )
        )
        fit, _ = analyze_fine_rabi(ds, node)

        figure = plot_fine_rabi(ds, [SimpleNamespace(name="q1")], True, "PI", fits=fit)

        data_axes = [axis for axis in figure.axes if axis.get_xlabel()]
        self.assertEqual(data_axes[0].get_title(), "q1: Fine Rabi scan")
        self.assertEqual(data_axes[0].get_xlabel(), "Pulse amplitude factor")
        self.assertEqual(data_axes[0].get_ylabel(), "Repetition group count")
        self.assertEqual(data_axes[1].get_title(), "q1: Fourier analysis")
        self.assertEqual(data_axes[1].get_ylabel(), "Frequency [cycles/group]")
        self.assertTrue(any("opt amp" in text.get_text() for text in data_axes[1].texts))
        fourier_lines = {
            line.get_label(): line
            for line in data_axes[1].lines
            if not line.get_label().startswith("_")
        }
        self.assertEqual(fourier_lines["Current amplitude"].get_color(), "tab:red")
        self.assertEqual(fourier_lines["Current amplitude"].get_linestyle(), "--")
        self.assertEqual(fourier_lines["Optimized amplitude"].get_color(), "tab:green")
        self.assertEqual(fourier_lines["Left linear fit"].get_color(), "tab:blue")
        self.assertEqual(fourier_lines["Right linear fit"].get_color(), "tab:blue")
        plt.close(figure)

    def test_branch_fit_intersection_finds_optimum(self):
        amp_prefactors = np.linspace(0.8, 1.2, 41)
        frequencies = np.fft.rfftfreq(81, d=1.0)
        branch_frequency = np.abs(amp_prefactors - 1.0)
        fourier = np.zeros((frequencies.size, amp_prefactors.size))
        for index, frequency in enumerate(branch_frequency):
            frequency_index = np.argmin(np.abs(frequencies - frequency))
            fourier[frequency_index, index] = 1.0

        result = fit_fourier_branches(amp_prefactors, frequencies, fourier)

        self.assertAlmostEqual(result["optimal_amp_prefactor"], 1.0, places=2)
        self.assertLess(abs(result["optimal_frequency"]), 0.02)

    def test_analysis_uses_fourier_branch_intersection(self):
        repetition_counts = np.arange(81)
        amp_prefactors = np.linspace(0.8, 1.2, 41)
        state = _make_v_ridge_state(
            repetition_counts,
            amp_prefactors,
            optimum=1.0,
            slope=1.0,
        )
        ds = xr.Dataset(
            {"state": (("qubit", "repetition_group_count", "amp_prefactor"), state[None, :, :])},
            coords={
                "qubit": ["q1"],
                "repetition_group_count": repetition_counts,
                "amp_prefactor": amp_prefactors,
            },
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                fourier_oversampling=8,
            )
        )

        fit, results = analyze_fine_rabi(ds, node)

        self.assertIn("fourier_amplitude", fit)
        self.assertIn("branch_line_coefficients", fit)
        self.assertEqual(fit.fourier_frequency.size, np.fft.rfftfreq(8 * repetition_counts.size).size)
        self.assertAlmostEqual(results["q1"]["optimal_amp_prefactor"], 1.0, places=2)

    def test_fourier_oversampling_must_be_positive(self):
        ds = xr.Dataset(
            {"state": (("qubit", "repetition_group_count", "amp_prefactor"), np.ones((1, 4, 3)))},
            coords={
                "qubit": ["q1"],
                "repetition_group_count": [0, 1, 2, 3],
                "amp_prefactor": [0.9, 1.0, 1.1],
            },
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                fourier_oversampling=0,
            )
        )

        with self.assertRaisesRegex(ValueError, "fourier_oversampling"):
            analyze_fine_rabi(ds, node)


def _make_v_ridge_state(repetition_counts, amp_prefactors, optimum, slope):
    frequencies = slope * np.abs(amp_prefactors - optimum)
    phase = 2 * np.pi * repetition_counts[:, None] * frequencies[None, :]
    return 0.5 + 0.35 * np.cos(phase)


if __name__ == "__main__":
    unittest.main()
