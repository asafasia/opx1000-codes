import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
from qm.exceptions import QMSimulationError

from utils.simulation import plot_waveform_report_safely, simulate_and_plot


class SimulationFallbackTests(unittest.TestCase):
    def test_sample_pull_failure_falls_back_to_waveform_report(self):
        waveform_report = Mock()
        job = Mock()
        job.get_simulated_samples.side_effect = QMSimulationError("sample pull failed")
        job.get_simulated_waveform_report.return_value = waveform_report
        qmm = Mock()
        qmm.simulate.return_value = job
        parameters = SimpleNamespace(
            simulation_duration_ns=1000,
            timeout=100,
            use_waveform_report=False,
        )

        with patch("utils.simulation.plt.gcf", return_value="fallback figure"):
            samples, figure, report = simulate_and_plot(qmm, {}, object(), parameters)

        job.wait_until.assert_called_once_with("Done", timeout=100)
        waveform_report.create_plot.assert_called_once_with(
            samples=None, plot=True, save_path=None
        )
        self.assertIsNone(samples)
        self.assertEqual(figure, "fallback figure")
        self.assertIs(report, waveform_report)

    def test_broken_waveform_report_renderer_does_not_crash(self):
        waveform_report = Mock()
        waveform_report.create_plot.side_effect = TypeError("unhashable type: 'list'")

        figure = plot_waveform_report_safely(waveform_report, samples=Mock())

        self.assertIn("could not render", figure.axes[0].texts[0].get_text())


if __name__ == "__main__":
    unittest.main()
