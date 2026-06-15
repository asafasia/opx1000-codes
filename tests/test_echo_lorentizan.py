import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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
            ),
            namespace={"qubits": [qubit]},
        )

        processed = lorentzian.process_raw_dataset(ds, node)

        np.testing.assert_allclose(processed.full_freq, [[4.099e9, 4.101e9]])
        np.testing.assert_allclose(processed.full_amp, [[0.06, 0.12]])

    def test_sequence_installs_waveform_pulse_and_sweeps_detuning_and_amplitude(self):
        source = (PROJECT_ROOT / "echo_lorentzian_sweep.py").read_text()

        self.assertIn("WaveformPulse(", source)
        self.assertIn("lorentzian_envelope(", source)
        self.assertIn("with for_(*from_array(df, dfs)):", source)
        self.assertIn("with for_(*from_array(a, amps)):", source)
        self.assertIn("qubit.xy.play(operation, amplitude_scale=a)", source)
        self.assertIn('"detuning": xr.DataArray(', source)
        self.assertIn('"amp_prefactor": xr.DataArray(', source)


if __name__ == "__main__":
    unittest.main()
