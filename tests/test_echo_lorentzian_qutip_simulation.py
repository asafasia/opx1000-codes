import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = REPOSITORY_ROOT / "Projects" / "echo-lorentzian-qutip-simulation"


def load_project_module(name: str):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parameters_module = load_project_module("parameters")
simulator = load_project_module("simulate_echo_lorentzian")


class EchoLorentzianQutipSimulationTests(unittest.TestCase):
    def test_waveform_parameters_match_echo_lorentzian_shapes(self):
        parameters = parameters_module.SimulationParameters(
            pulse_shape="root_lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
            echo=True,
        )

        waveform = simulator.build_waveform(parameters)

        self.assertEqual(len(waveform), 9)
        self.assertAlmostEqual(abs(waveform[0]), 0.05)
        self.assertAlmostEqual(abs(waveform[4]), 0.2)
        self.assertTrue(np.all(waveform[:4] > 0))
        self.assertTrue(np.all(waveform[4:] < 0))

    def test_sweep_axes_use_same_amplitude_and_detuning_parameters(self):
        parameters = parameters_module.SimulationParameters(
            min_amp_factor=0.0,
            max_amp_factor=1.0,
            amp_factor_step=0.25,
            frequency_span_in_mhz=10,
            frequency_step_in_mhz=5,
        )

        detunings, amps = simulator.sweep_axes(parameters)

        np.testing.assert_allclose(amps, [0.0, 0.25, 0.5, 0.75])
        np.testing.assert_allclose(detunings, [-5e6, 0, 5e6])

    def test_stretched_waveform_keeps_requested_physical_length(self):
        parameters = parameters_module.SimulationParameters(
            pulse_shape="gaussian",
            lorentzian_length_in_ns=12,
            waveform_template_length_in_ns=6,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
        )

        waveform = simulator.stretched_waveform(parameters)

        self.assertEqual(len(waveform), 12)

    def test_simulator_source_uses_qutip_mesolve(self):
        source = (PROJECT_ROOT / "simulate_echo_lorentzian.py").read_text()

        self.assertIn("import qutip", source)
        self.assertIn("qutip.mesolve", source)


if __name__ == "__main__":
    unittest.main()
