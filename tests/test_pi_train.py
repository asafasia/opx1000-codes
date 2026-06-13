import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.pi_train.plotting import plot_pi_train


REPOSITORY_ROOT = Path(__file__).parent.parent


class PiTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (REPOSITORY_ROOT / "calibrations" / "04c_pi_train.py").read_text()

    def test_sequence_sweeps_zero_through_maximum_selected_gate_count(self):
        self.assertIn("np.arange(node.parameters.max_number_of_pulses + 1", self.source)
        self.assertIn("with for_(count, 0, count < pulse_count, count + 1):", self.source)
        self.assertIn("operation = node.parameters.operation", self.source)
        self.assertIn("qubit.xy.play(operation)", self.source)
        self.assertIn("qubit.reset(", self.source)

    def test_parameters_offer_pi_and_pi_over_two_gates(self):
        source = (
            REPOSITORY_ROOT / "calibration_utils" / "pi_train" / "parameters.py"
        ).read_text()
        self.assertIn('operation: Literal["x180", "x90"] = "x180"', source)

    def test_sequence_supports_state_and_iq_readout(self):
        self.assertIn("if node.parameters.use_state_discrimination:", self.source)
        self.assertIn("qubit.readout_state(state[i])", self.source)
        self.assertIn('save(f"state{i + 1}")', self.source)
        self.assertIn('save(f"I{i + 1}")', self.source)
        self.assertIn('save(f"Q{i + 1}")', self.source)

    def test_state_plot_shows_alternating_measured_state(self):
        expected = np.array([[0.0, 1.0, 0.0, 1.0]])
        ds = xr.Dataset(
            {"state": (("qubit", "number_of_pulses"), expected)},
            coords={"qubit": ["q1"], "number_of_pulses": [0, 1, 2, 3]},
        )

        figure = plot_pi_train(ds, [SimpleNamespace(name="q1")], True, "x180")

        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), expected[0])
        self.assertEqual(figure.axes[0].get_ylabel(), "Excited-state population")
        self.assertEqual(figure.axes[0].get_xlabel(), "Number of consecutive x180 gates")
        plt.close(figure)

    def test_pi_over_two_plot_labels_selected_gate(self):
        ds = xr.Dataset(
            {"state": (("qubit", "number_of_pulses"), [[0.0, 0.5, 1.0, 0.5, 0.0]])},
            coords={"qubit": ["q1"], "number_of_pulses": [0, 1, 2, 3, 4]},
        )

        figure = plot_pi_train(ds, [SimpleNamespace(name="q1")], True, "x90")

        self.assertEqual(figure.axes[0].get_xlabel(), "Number of consecutive x90 gates")
        self.assertEqual(figure._suptitle.get_text(), "Gate train: x90")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
