import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = (
    REPOSITORY_ROOT / "Projects" / "shaped_pulse_spectroscopy" / "simulation" / "qutip"
)


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
cutoff_simulator = load_project_module("simulate_cutoff_amp_fwhm_map")


class EchoLorentzianQutipSimulationTests(unittest.TestCase):
    def test_cutoff_simulation_uses_log_grid_like_experiment(self):
        cutoffs = cutoff_simulator.cutoff_values(0.01, 0.99, 3)

        np.testing.assert_allclose(cutoffs, [0.99, np.sqrt(0.0099), 0.01])

    def test_default_physical_parameters_match_single_qubit_q1(self):
        parameters = parameters_module.SimulationParameters()

        self.assertEqual(parameters.qubit_name, "q1")
        self.assertAlmostEqual(parameters.rf_frequency_hz, 4267100311.915768)
        self.assertAlmostEqual(parameters.x180_amplitude, 0.15127819777954318)
        self.assertEqual(parameters.x180_length_in_ns, 40.0)
        self.assertAlmostEqual(parameters.lorentzian_peak_amplitude, 0.15127819777954318)
        self.assertAlmostEqual(parameters.t1_in_us, 30.87)
        self.assertAlmostEqual(parameters.t2_star_in_us, 6.855588134130561)

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

    def test_standard_lorentzian_uses_requested_cutoff(self):
        parameters = parameters_module.SimulationParameters(
            pulse_shape="lorentzian",
            lorentzian_length_in_ns=9,
            lorentzian_peak_amplitude=0.2,
            cutoff=0.25,
        )

        waveform = simulator.build_waveform(parameters)

        self.assertAlmostEqual(waveform[0], 0.05)
        self.assertAlmostEqual(waveform[-1], 0.05)
        self.assertAlmostEqual(waveform[4], 0.2)

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

    def test_sweep_axes_support_experimental_log_and_point_grids(self):
        parameters = parameters_module.SimulationParameters(
            min_amp_factor=0.01,
            max_amp_factor=1.0,
            amp_factor_points=3,
            amp_factor_spacing="log",
            frequency_span_in_mhz=1,
            frequency_points=5,
        )

        detunings, amps = simulator.sweep_axes(parameters)

        np.testing.assert_allclose(amps, [0.01, 0.1, 1.0])
        np.testing.assert_allclose(detunings, [-0.5e6, -0.25e6, 0, 0.25e6, 0.5e6])

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

    def test_simulation_time_axis_matches_reference_solver(self):
        parameters = parameters_module.SimulationParameters(
            lorentzian_length_in_ns=40,
            num_time_points=5,
        )

        times = simulator.simulation_time_axis(parameters)

        np.testing.assert_allclose(times * 1e9, [-20, -10, 0, 10, 20])

    def test_simulation_time_axis_rejects_too_few_points(self):
        parameters = parameters_module.SimulationParameters(num_time_points=1)

        with self.assertRaisesRegex(ValueError, "num_time_points"):
            simulator.simulation_time_axis(parameters)

    def test_simulator_source_uses_qutip_mesolve(self):
        source = (PROJECT_ROOT / "simulate_echo_lorentzian.py").read_text()

        self.assertIn("import qutip", source)
        self.assertIn("qutip.mesolve", source)
        self.assertIn("qutip.destroy(2)", source)
        self.assertIn("simulate_point_populations", source)
        self.assertIn("anharmonicity_hz", source)
        self.assertIn("qutip.qeye(num_levels)", source)
        self.assertIn(
            "from shaped_pulse_spectroscopy.fwhm import add_gaussian_fwhm_analysis",
            source,
        )
        self.assertIn("save_fwhm_results", source)
        self.assertIn("T2* limit: +/- 1/(2*pi*T2*)", source)
        self.assertIn("1 / (2 * np.pi * t2_star_s)", source)


if __name__ == "__main__":
    unittest.main()
