import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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
                root_lorentzian_cutoff=0.25,
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

    def test_sequence_installs_waveform_pulse_and_sweeps_detuning_and_amplitude(self):
        source = (PROJECT_ROOT / "echo_lorentzian_sweep.py").read_text()
        v2_source = (PROJECT_ROOT / "echo_lorentzian_v2.py").read_text()

        self.assertIn("install_lorentzian_operation(node)", source)
        self.assertIn("class EchoLorentzian(BaseCalibration", v2_source)
        self.assertIn("install_lorentzian_operation(self)", v2_source)
        self.assertIn("with for_(*from_array(df, dfs)):", source)
        self.assertIn("with for_(*from_array(a, amps)):", source)
        self.assertIn("qubit.xy.play(operation, amplitude_scale=a)", source)
        self.assertIn('"detuning": xr.DataArray(', source)
        self.assertIn('"amp_prefactor": xr.DataArray(', source)

    def test_shared_operation_builder_supports_root_lorentzian(self):
        parameters = SimpleNamespace(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_tau_in_ns=2,
            lorentzian_peak_amplitude=0.2,
            root_lorentzian_cutoff=0.25,
            echo=False,
        )

        waveform = lorentzian.build_waveform(parameters)

        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)

    def test_echo_phase_jump_flips_second_half_of_waveform(self):
        parameters = SimpleNamespace(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_tau_in_ns=2,
            lorentzian_peak_amplitude=0.2,
            root_lorentzian_cutoff=0.25,
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
        ds.attrs.update(
            {
                "pulse_shape": "root_lorentzian",
                "lorentzian_length_in_ns": 80,
                "lorentzian_peak_amplitude": 0.12,
                "root_lorentzian_cutoff": 0.25,
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
        self.assertIn("Rabi frequency [Hz]", ylabels)
        self.assertIn("RF frequency [GHz]", secondary_labels)
        self.assertIn("Lorentzian peak amplitude [V]", secondary_ylabels)
        figure_text = " ".join(text.get_text() for text in figure.texts)
        self.assertIn("root_lorentzian", figure_text)
        self.assertIn("80 ns", figure_text)
        self.assertIn("echo=True", figure_text)
        self.assertIn("0.25 cutoff", figure_text)
        self.assertIn("T1=2 us", figure_text)
        self.assertIn("T2=1 us", figure_text)
        self.assertIn("1/(pi*T2)=318310", figure_text)
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
        plt.close(figure)

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
