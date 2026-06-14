import unittest
from types import SimpleNamespace

import xarray as xr

from utils.plotting_settings import plot_per_qubit, qubit_grid_locations


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


if __name__ == "__main__":
    unittest.main()
