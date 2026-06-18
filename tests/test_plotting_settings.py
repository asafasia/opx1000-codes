import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from calibration_utils.resonator_spectroscopy.plotting import plot_raw_amplitude
from utils.plotting_settings import (
    CALIBRATION_PARAMETERS_GID,
    CalibrationPlot,
    format_readout_parameter_lines,
    plot_per_qubit,
    qubit_grid_locations,
)


class PlottingSettingsTests(unittest.TestCase):
    def test_plot_per_qubit_calls_plotter_with_only_one_qubit(self):
        calls = []

        def plotter(dataset, qubits, suffix):
            calls.append((dataset, [qubit.name for qubit in qubits], suffix))
            return f"figure-{qubits[0].name}"

        figures = plot_per_qubit(
            plotter,
            "dataset",
            [SimpleNamespace(name="q1"), SimpleNamespace(name="q2")],
            "fit",
            figure_name="amplitude",
        )

        self.assertEqual(
            figures,
            {"amplitude_q1": "figure-q1", "amplitude_q2": "figure-q2"},
        )
        self.assertEqual(calls, [("dataset", ["q1"], "fit"), ("dataset", ["q2"], "fit")])

    def test_single_qubit_grid_location_is_compact(self):
        qubits = [SimpleNamespace(grid_location="4,3")]

        self.assertEqual(qubit_grid_locations(qubits), ["0,0"])

    def test_calibration_plot_adds_standard_parameter_box(self):
        figure = plt.figure()

        text = CalibrationPlot(figure).add_parameters(["Parameters", "num reps=10"])

        self.assertEqual(text.get_gid(), CALIBRATION_PARAMETERS_GID)
        self.assertEqual(text.get_text(), "Parameters\nnum reps=10")
        self.assertEqual(text.get_family()[0], "monospace")
        self.assertEqual(text.get_bbox_patch().get_boxstyle().__class__.__name__, "Round")
        plt.close(figure)

    def test_format_readout_parameter_lines_uses_common_qubit_fields(self):
        qubit = SimpleNamespace(
            name="q1",
            resonator=SimpleNamespace(
                operations={"readout": SimpleNamespace(length=800, amplitude=0.032)}
            ),
        )

        self.assertEqual(
            format_readout_parameter_lines([qubit]),
            ["q1: readout length=800 ns, readout amp=0.032 V"],
        )

    def test_plot_per_qubit_slices_dataset_and_fit(self):
        dataset = xr.Dataset({"I": ("qubit", [1.0, 2.0])}, coords={"qubit": ["q1", "q2"]})
        fit = xr.Dataset({"frequency": ("qubit", [3.0, 4.0])}, coords={"qubit": ["q1", "q2"]})

        figures = plot_per_qubit(
            lambda selected, qubits, selected_fit: (
                list(selected.qubit.values),
                list(selected_fit.qubit.values),
            ),
            dataset,
            [SimpleNamespace(name="q1"), SimpleNamespace(name="q2")],
            fit,
            figure_name="result",
        )

        self.assertEqual(figures["result_q1"], (["q1"], ["q1"]))
        self.assertEqual(figures["result_q2"], (["q2"], ["q2"]))

    def test_resonator_spectroscopy_creates_one_figure_per_qubit(self):
        detuning = np.array([-1e6, 0.0, 1e6])
        dataset = xr.Dataset(
            {
                "full_freq": (
                    ("qubit", "detuning"),
                    [[6.9e9, 6.901e9, 6.902e9], [7.1e9, 7.101e9, 7.102e9]],
                ),
                "ground_IQ_abs": (("qubit", "detuning"), [[1e-3, 2e-3, 1e-3], [2e-3, 3e-3, 2e-3]]),
                "mixed_IQ_abs": (("qubit", "detuning"), [[2e-3, 1e-3, 2e-3], [3e-3, 2e-3, 3e-3]]),
                "IQ_separation": (("qubit", "detuning"), [[1.0, 3.0, 1.0], [1.0, 4.0, 1.0]]),
            },
            coords={"qubit": ["q1", "q2"], "detuning": detuning},
        )
        qubits = [
            SimpleNamespace(name="q1", grid_location="0,0", resonator=SimpleNamespace(RF_frequency=6.901e9)),
            SimpleNamespace(name="q2", grid_location="1,4", resonator=SimpleNamespace(RF_frequency=7.101e9)),
        ]

        figures = plot_per_qubit(
            plot_raw_amplitude,
            dataset,
            qubits,
            figure_name="resonator_spectroscopy",
        )

        self.assertEqual(set(figures), {"resonator_spectroscopy_q1", "resonator_spectroscopy_q2"})
        for qubit_name, figure in (("q1", figures["resonator_spectroscopy_q1"]), ("q2", figures["resonator_spectroscopy_q2"])):
            titles = [axis.get_title() for axis in figure.axes]
            self.assertIn(qubit_name, titles)
            self.assertNotIn("q2" if qubit_name == "q1" else "q1", titles)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
