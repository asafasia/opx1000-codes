import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


REPOSITORY_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = REPOSITORY_ROOT / "Projects" / "echo-lorentizan"


def load_project_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lorentzian = load_project_module("lorentzian")


def make_plot_qubit(name="q7", t1=None, t2_ramsey=None):
    return SimpleNamespace(
        name=name,
        T1=t1,
        t2_ramsey=t2_ramsey,
        T2ramsey=t2_ramsey,
        xy=SimpleNamespace(
            operations={
                "x180": SimpleNamespace(amplitude=0.1, length=40),
            }
        ),
    )


class EchoLorentizanTests(unittest.TestCase):
    def test_lorentzian_envelope_is_centered_and_user_length(self):
        waveform = lorentzian.lorentzian_envelope(
            length_ns=9,
            tau_ns=2,
            peak_amplitude=0.2,
        )

        self.assertEqual(len(waveform), 9)
        self.assertAlmostEqual(waveform[4], 0.2)
        np.testing.assert_allclose(waveform, waveform[::-1])
        self.assertLess(waveform[0], waveform[4])

    def test_lorentzian_envelope_validates_user_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            lorentzian.lorentzian_envelope(3, 2, 0.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            lorentzian.lorentzian_envelope(8, 0, 0.1)

    def test_root_lorentzian_envelope_derives_tau_from_cutoff(self):
        waveform = lorentzian.root_lorentzian_envelope(
            length_ns=9,
            cutoff=0.25,
            peak_amplitude=0.2,
        )

        self.assertEqual(len(waveform), 9)
        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[-1], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)
        np.testing.assert_allclose(waveform, waveform[::-1])

    def test_root_lorentzian_envelope_validates_cutoff(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            lorentzian.root_lorentzian_envelope(3, 0.5, 0.1)
        with self.assertRaisesRegex(ValueError, "0 < cutoff <= 1"):
            lorentzian.root_lorentzian_envelope(8, 0, 0.1)
        with self.assertRaisesRegex(ValueError, "0 < cutoff <= 1"):
            lorentzian.root_lorentzian_envelope(8, 1.1, 0.1)
        self.assertEqual(
            lorentzian.root_lorentzian_envelope(4, 1, 0.1),
            [0.1, 0.1, 0.1, 0.1],
        )

    def test_gaussian_envelope_derives_sigma_from_cutoff(self):
        waveform = lorentzian.gaussian_envelope(
            length_ns=9,
            cutoff=0.25,
            peak_amplitude=0.2,
        )

        self.assertEqual(len(waveform), 9)
        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[-1], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)
        np.testing.assert_allclose(waveform, waveform[::-1])

    def test_gaussian_envelope_validates_cutoff(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            lorentzian.gaussian_envelope(3, 0.5, 0.1)
        with self.assertRaisesRegex(ValueError, "0 < cutoff <= 1"):
            lorentzian.gaussian_envelope(8, 0, 0.1)
        with self.assertRaisesRegex(ValueError, "0 < cutoff <= 1"):
            lorentzian.gaussian_envelope(8, 1.1, 0.1)
        self.assertEqual(
            lorentzian.gaussian_envelope(4, 1, 0.1),
            [0.1, 0.1, 0.1, 0.1],
        )

    def test_analysis_adds_frequency_and_lorentzian_peak_amplitude(self):
        ds = xr.Dataset(
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
        qubit = SimpleNamespace(xy=SimpleNamespace(RF_frequency=4.1e9))
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                lorentzian_peak_amplitude=0.12,
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=80,
                lorentzian_tau_in_ns=8.0,
                cutoff=0.25,
                echo=True,
                min_amp_factor=0.0,
                max_amp_factor=2.0,
                amp_factor_step=0.03,
                frequency_span_in_mhz=100,
                frequency_step_in_mhz=2,
            ),
            namespace={"qubits": [qubit]},
        )

        processed = lorentzian.process_raw_dataset(ds, node)

        np.testing.assert_allclose(processed.full_freq, [[4.099e9, 4.101e9]])
        np.testing.assert_allclose(processed.full_amp, [[0.06, 0.12]])
        self.assertEqual(processed.attrs["pulse_shape"], "root_lorentzian")
        self.assertEqual(processed.attrs["lorentzian_length_in_ns"], 80)
        self.assertTrue(processed.attrs["echo"])

    def test_analysis_adds_gaussian_fwhm_per_amplitude(self):
        detuning = np.linspace(-4e6, 4e6, 81)
        amps = [0.5, 1.0]
        sigma = 0.4e6
        centers = [-0.5e6, 1.0e6]
        traces = [
            0.1 + 0.8 * np.exp(-0.5 * ((detuning - center) / sigma) ** 2)
            for center in centers
        ]
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "detuning", "amp_prefactor"),
                    np.array(traces).T[np.newaxis, :, :],
                )
            },
            coords={
                "qubit": ["q7"],
                "detuning": detuning,
                "amp_prefactor": amps,
            },
        )
        qubit = SimpleNamespace(xy=SimpleNamespace(RF_frequency=4.1e9))
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                lorentzian_peak_amplitude=0.12,
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=80,
                lorentzian_tau_in_ns=8.0,
                cutoff=0.25,
                echo=False,
                min_amp_factor=0.0,
                max_amp_factor=2.0,
                amp_factor_step=0.03,
                frequency_span_in_mhz=100,
                frequency_step_in_mhz=2,
            ),
            namespace={"qubits": [qubit]},
        )

        processed = lorentzian.process_raw_dataset(ds, node)

        expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        np.testing.assert_allclose(
            processed.gaussian_center_hz.sel(qubit="q7"),
            centers,
            rtol=0,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            processed.gaussian_fwhm_hz.sel(qubit="q7"),
            expected_fwhm,
            rtol=1e-6,
        )
        self.assertTrue(
            np.all(processed.gaussian_fit_r_squared.sel(qubit="q7").values > 0.99)
        )

    def test_gaussian_fwhm_rejects_high_detuning_center(self):
        detuning = np.linspace(-4e6, 4e6, 81)
        sigma = 0.5e6
        state = 0.1 + 0.8 * np.exp(-0.5 * ((detuning - 3.8e6) / sigma) ** 2)
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "detuning", "amp_prefactor"),
                    state[np.newaxis, :, np.newaxis],
                )
            },
            coords={
                "qubit": ["q7"],
                "detuning": detuning,
                "amp_prefactor": [0.5],
            },
        )

        processed = lorentzian.add_gaussian_fwhm_analysis(
            ds,
            use_state_discrimination=True,
        )

        self.assertTrue(np.isnan(float(processed.gaussian_center_hz.values[0, 0])))
        self.assertTrue(np.isnan(float(processed.gaussian_fwhm_hz.values[0, 0])))

    def test_sequence_installs_waveform_pulse_and_sweeps_detuning_and_amplitude(self):
        source = (PROJECT_ROOT / "echo_lorentzian_sweep.py").read_text()
        v2_source = (PROJECT_ROOT / "echo_lorentzian_v2.py").read_text()
        amplitude_source = (PROJECT_ROOT / "echo_lorentzian_amplitude_v2.py").read_text()

        self.assertIn("install_lorentzian_operation(node)", source)
        self.assertIn("class EchoLorentzian(BaseCalibration", v2_source)
        self.assertIn("install_lorentzian_operation(self)", v2_source)
        self.assertIn("class EchoLorentzianAmplitude(BaseCalibration", amplitude_source)
        self.assertIn("with for_(*from_array(a, amps)):", amplitude_source)
        self.assertNotIn("with for_(*from_array(df, dfs)):", amplitude_source)
        self.assertIn("with for_(*from_array(df, dfs)):", source)
        self.assertIn("with for_(*from_array(a, amps)):", source)
        self.assertIn("duration=play_duration", source)
        self.assertIn("duration=play_duration", v2_source)
        self.assertIn("duration=play_duration", amplitude_source)
        self.assertIn('"detuning": xr.DataArray(', source)
        self.assertIn('"amp_prefactor": xr.DataArray(', source)

    def test_shared_operation_builder_supports_root_lorentzian(self):
        parameters = SimpleNamespace(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_tau_in_ns=2,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
            echo=False,
        )

        waveform = lorentzian.build_waveform(parameters)

        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)

    def test_shared_operation_builder_supports_gaussian(self):
        parameters = SimpleNamespace(
            pulse_shape="gaussian",
            lorentzian_length_in_ns=9,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
            echo=False,
        )

        waveform = lorentzian.build_waveform(parameters)

        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)

    def test_waveform_template_stretches_to_long_play_duration(self):
        qubit = SimpleNamespace(
            xy=SimpleNamespace(operations={}),
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=80000,
                waveform_template_length_in_ns=2000,
                lorentzian_tau_in_ns=2,
                lorentzian_peak_amplitude=0.1,
                cutoff=0.25,
                echo=False,
                operation="lorentzian",
                min_amp_factor=0.0,
                max_amp_factor=1.0,
                amp_factor_step=0.1,
            ),
            namespace={"qubits": [qubit]},
        )

        waveform = lorentzian.install_lorentzian_operation(node)

        self.assertEqual(len(waveform), 2000)
        self.assertEqual(qubit.xy.operations["lorentzian"].length, 2000)
        self.assertEqual(node.namespace["lorentzian_play_duration_cycles"], 20000)

    def test_lorentzian_safety_limit_allows_samples_below_one_volt(self):
        qubit = SimpleNamespace(
            xy=SimpleNamespace(operations={}),
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=5000,
                waveform_template_length_in_ns=5000,
                lorentzian_tau_in_ns=2,
                lorentzian_peak_amplitude=0.8,
                cutoff=0.99,
                echo=False,
                operation="lorentzian",
                min_amp_factor=0.0,
                max_amp_factor=1.0,
                amp_factor_step=0.01,
            ),
            namespace={"qubits": [qubit]},
        )

        waveform = lorentzian.install_lorentzian_operation(node)

        self.assertLess(max(abs(sample) for sample in waveform) * 0.99, 1.0)

    def test_lorentzian_safety_limit_rejects_samples_at_one_volt(self):
        qubit = SimpleNamespace(
            xy=SimpleNamespace(operations={}),
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=20,
                waveform_template_length_in_ns=20,
                lorentzian_tau_in_ns=2,
                lorentzian_peak_amplitude=1.0,
                cutoff=1.0,
                echo=False,
                operation="lorentzian",
                min_amp_factor=1.0,
                max_amp_factor=1.01,
                amp_factor_step=0.01,
            ),
            namespace={"qubits": [qubit]},
        )

        with self.assertRaisesRegex(ValueError, "below 1 V"):
            lorentzian.install_lorentzian_operation(node)

    def test_waveform_template_requires_play_length_divisible_by_four(self):
        parameters = SimpleNamespace(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=80002,
            waveform_template_length_in_ns=2000,
            lorentzian_tau_in_ns=2,
            lorentzian_peak_amplitude=0.1,
            cutoff=0.25,
            echo=False,
        )

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            lorentzian.lorentzian_play_duration_cycles(parameters)

    def test_echo_phase_jump_flips_second_half_of_waveform(self):
        parameters = SimpleNamespace(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_tau_in_ns=2,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
            echo=True,
        )

        waveform = np.array(lorentzian.build_waveform(parameters))
        magnitude = np.array(
            lorentzian.root_lorentzian_envelope(
                length_ns=9,
                cutoff=0.25,
                peak_amplitude=0.2,
            )
        )

        self.assertTrue(np.all(waveform[:4] > 0))
        self.assertTrue(np.all(waveform[4:] < 0))
        np.testing.assert_allclose(np.abs(waveform), magnitude)

    def test_lorentzian_plot_uses_detuning_bottom_and_absolute_frequency_top(self):
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "detuning", "amp_prefactor"),
                    np.zeros((1, 2, 3)),
                )
            },
            coords={
                "qubit": ["q7"],
                "detuning": [-1e6, 1e6],
                "amp_prefactor": [0.5, 1.0, 1.5],
                "full_freq": (("qubit", "detuning"), [[4.099e9, 4.101e9]]),
                "full_amp": (("qubit", "amp_prefactor"), [[0.06, 0.12, 0.18]]),
            },
        )
        ds.attrs.update(
            {
                "pulse_shape": "root_lorentzian",
                "lorentzian_length_in_ns": 80,
                "lorentzian_peak_amplitude": 0.12,
                "cutoff": 0.25,
                "echo": True,
                "frequency_span_in_mhz": 100,
                "frequency_step_in_mhz": 2,
            }
        )
        qubit = make_plot_qubit(t1=2e-6, t2_ramsey=1e-6)

        figure = lorentzian.plot_raw_data(
            ds,
            [qubit],
            use_state_discrimination=True,
        )

        labels = {axis.get_xlabel() for axis in figure.axes}
        ylabels = {axis.get_ylabel() for axis in figure.axes}
        secondary_labels = {
            child.get_xlabel()
            for axis in figure.axes
            for child in axis.get_children()
            if type(child).__name__ == "SecondaryAxis"
        }
        secondary_ylabels = {
            child.get_ylabel()
            for axis in figure.axes
            for child in axis.get_children()
            if type(child).__name__ == "SecondaryAxis"
        }
        self.assertIn("Detuning [MHz]", labels)
        self.assertIn("Rabi frequency [MHz]", ylabels)
        self.assertIn("RF frequency [GHz]", secondary_labels)
        self.assertIn("Lorentzian peak amplitude [V]", secondary_ylabels)
        self.assertEqual(figure.axes[0].collections[0].get_array().shape, (3, 2))
        figure_text = " ".join(text.get_text() for text in figure.texts)
        self.assertIn("root_lorentzian", figure_text)
        self.assertIn("80 ns", figure_text)
        self.assertIn("echo=True", figure_text)
        self.assertIn("0.25 cutoff", figure_text)
        self.assertIn("T1=2 us", figure_text)
        self.assertIn("T2=1 us", figure_text)
        self.assertIn("1/(pi*T2)=318310", figure_text)
        plt.close(figure)

    def test_lorentzian_plot_marks_gaussian_fwhm_edges(self):
        detuning = np.linspace(-4e6, 4e6, 81)
        amps = [0.5, 1.0]
        sigma = 0.8e6
        state = 0.1 + 0.8 * np.exp(-0.5 * (detuning / sigma) ** 2)
        states = np.stack([state, state], axis=-1)
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "detuning", "amp_prefactor"),
                    states[np.newaxis, :, :],
                )
            },
            coords={
                "qubit": ["q7"],
                "detuning": detuning,
                "amp_prefactor": amps,
            },
        )
        qubit = SimpleNamespace(xy=SimpleNamespace(RF_frequency=4.1e9))
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                lorentzian_peak_amplitude=0.12,
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=80,
                lorentzian_tau_in_ns=8.0,
                cutoff=0.25,
                echo=False,
                min_amp_factor=0.0,
                max_amp_factor=2.0,
                amp_factor_step=0.03,
                frequency_span_in_mhz=100,
                frequency_step_in_mhz=2,
            ),
            namespace={"qubits": [qubit]},
        )
        processed = lorentzian.process_raw_dataset(ds, node)
        plot_qubit = make_plot_qubit()

        figure = lorentzian.plot_raw_data(
            processed,
            [plot_qubit],
            use_state_discrimination=True,
        )

        legend_text = " ".join(
            text.get_text()
            for axis in figure.axes
            for legend in [axis.get_legend()]
            if legend is not None
            for text in legend.get_texts()
        )
        self.assertIn("Gaussian FWHM", legend_text)
        plt.close(figure)

    def test_amplitude_sweep_processing_and_plot_are_1d(self):
        ds = xr.Dataset(
            {
                "state": (
                    ("qubit", "amp_prefactor"),
                    np.zeros((1, 2)),
                )
            },
            coords={
                "qubit": ["q7"],
                "amp_prefactor": [0.5, 1.0],
            },
        )
        qubit = SimpleNamespace(
            xy=SimpleNamespace(
                RF_frequency=4.1e9,
                operations={"x180": SimpleNamespace(amplitude=0.1, length=40)},
            ),
            name="q7",
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(
                use_state_discrimination=True,
                lorentzian_peak_amplitude=0.12,
                pulse_shape="root_lorentzian",
                lorentzian_length_in_ns=80,
                lorentzian_tau_in_ns=8.0,
                cutoff=0.25,
                echo=False,
                min_amp_factor=0.0,
                max_amp_factor=2.0,
                amp_factor_step=0.03,
                frequency_span_in_mhz=100,
                frequency_step_in_mhz=2,
            ),
            namespace={"qubits": [qubit]},
        )

        processed = lorentzian.process_amplitude_dataset(ds, node)
        figure = lorentzian.plot_amplitude_sweep(
            processed,
            [qubit],
            use_state_discrimination=True,
        )

        np.testing.assert_allclose(processed.full_amp, [[0.06, 0.12]])
        np.testing.assert_allclose(processed.full_freq, [4.1e9])
        self.assertNotIn("detuning", processed.coords)
        self.assertEqual(processed.attrs["detuning_hz"], 0)
        self.assertEqual(figure.axes[0].get_xlabel(), "Rabi frequency [MHz]")
        secondary_labels = {
            child.get_xlabel()
            for axis in figure.axes
            for child in axis.get_children()
            if type(child).__name__ == "SecondaryAxis"
        }
        self.assertIn("Lorentzian peak amplitude [V]", secondary_labels)
        plt.close(figure)

    def test_lorentzian_plot_marks_t2_limit_lines(self):
        ds = xr.Dataset(
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
                "full_freq": (("qubit", "detuning"), [[4.099e9, 4.101e9]]),
                "full_amp": (("qubit", "amp_prefactor"), [[0.06, 0.12]]),
            },
        )
        qubit = make_plot_qubit(t2_ramsey=1e-6)

        figure = lorentzian.plot_raw_data(
            ds,
            [qubit],
            use_state_discrimination=True,
        )

        t2_line_positions = sorted(
            float(line.get_xdata()[0])
            for line in figure.axes[0].lines
            if len(line.get_xdata()) == 2
        )
        expected_limit_mhz = 1 / (2 * np.pi * qubit.T2ramsey) / 1e6
        np.testing.assert_allclose(
            t2_line_positions,
            [-expected_limit_mhz, expected_limit_mhz],
        )
        labels = [line.get_label() for line in figure.axes[0].lines]
        self.assertIn("T2 limit: ±1/(2πT2)", labels)
        plt.close(figure)

    def test_t2_ramsey_ns_attribute_is_interpreted_as_seconds(self):
        qubit = make_plot_qubit(t2_ramsey=None)
        qubit.T2ramsey = None
        qubit.t2_ramsey_ns = 1e-6

        self.assertAlmostEqual(lorentzian._t2_seconds(qubit), 1e-6)
        self.assertAlmostEqual(lorentzian._t2_limit_hz(qubit), 1 / (np.pi * 1e-6))

    def test_profile_metrics_override_quam_default_coherence(self):
        qubit = make_plot_qubit(name="q9", t1=1e-6, t2_ramsey=None)
        qubit.T2ramsey = None
        profile = {
            "metrics": {
                "qubits": {
                    "q9": {
                        "coherence": {
                            "t1_ns": 38823.217302558165,
                            "t2_ramsey_ns": 4.0462030446001355e-06,
                            "t2_echo_ns": None,
                        }
                    }
                }
            }
        }

        with patch.object(lorentzian, "current_profile_name", return_value="single_qubit"), patch.object(
            lorentzian, "load_profile", return_value=profile
        ):
            self.assertAlmostEqual(lorentzian._t1_seconds(qubit), 38.82321730255817e-6)
            self.assertAlmostEqual(lorentzian._t2_seconds(qubit), 4.0462030446001355e-6)

    def test_amplitude_to_rabi_frequency_uses_square_pi_pulse(self):
        from utils.rabi_amplitude import amplitude_to_rabi_frequency_hz

        self.assertAlmostEqual(
            amplitude_to_rabi_frequency_hz(
                general_amp=0.05,
                pi_amp=0.1,
                pi_length_ns=40,
            ),
            6.25e6,
        )


if __name__ == "__main__":
    unittest.main()
