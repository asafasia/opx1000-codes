import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from calibration_utils.power_rabi.plotting import plot_raw_data_with_fit


REPOSITORY_ROOT = Path(__file__).parent.parent


class RabiStateDiscriminationTests(unittest.TestCase):
    def test_power_rabi_uses_parameter_for_acquisition_and_plotting(self):
        source = (REPOSITORY_ROOT / "calibrations" / "04b_power_rabi.py").read_text()

        self.assertNotIn("node.use_state_discrimination", source)
        self.assertIn("if node.parameters.use_state_discrimination:", source)
        self.assertIn("qubit.readout_state(state[i])", source)
        self.assertIn('save(f"state{i + 1}")', source)
        self.assertIn("validate_readout_dataset(dataset, node.parameters.use_state_discrimination)", source)
        self.assertIn("use_state_discrimination=node.parameters.use_state_discrimination", source)

    def test_rabi_chevron_uses_parameter_for_acquisition_and_plotting(self):
        source = (REPOSITORY_ROOT / "calibrations" / "04a_rabi_chevron.py").read_text()

        self.assertIn("state_discrimination = node.parameters.use_state_discrimination", source)
        self.assertIn("state = [declare(int) for _ in range(num_qubits)]", source)
        self.assertIn("qubit.readout_state(state[i])", source)
        self.assertIn('save(f"state{i + 1}")', source)
        self.assertIn("validate_readout_dataset(dataset, node.parameters.use_state_discrimination)", source)
        self.assertIn("use_state_discrimination=node.parameters.use_state_discrimination", source)

    def test_power_rabi_state_mode_plots_only_state_even_if_iq_are_present(self):
        coords = {
            "qubit": ["q9"],
            "nb_of_pulses": [1, 3],
            "amp_prefactor": [0.8, 1.0, 1.2],
        }
        shape = (1, 2, 3)
        ds = xr.Dataset(
            {
                "state": (("qubit", "nb_of_pulses", "amp_prefactor"), np.zeros(shape)),
                "I": (("qubit", "nb_of_pulses", "amp_prefactor"), np.ones(shape)),
                "Q": (("qubit", "nb_of_pulses", "amp_prefactor"), np.ones(shape)),
            },
            coords=coords,
        ).assign_coords(full_amp=(("qubit", "amp_prefactor"), [[0.08, 0.1, 0.12]]))
        fits = ds.assign(
            success=("qubit", [True]),
            opt_amp=("qubit", [0.1]),
        )

        figure = plot_raw_data_with_fit(
            ds,
            [SimpleNamespace(name="q9")],
            fits,
            use_state_discrimination=True,
        )

        titled_axes = [axis for axis in figure.axes if axis.get_title()]
        self.assertEqual([axis.get_title() for axis in titled_axes], ["q9: state"])


if __name__ == "__main__":
    unittest.main()
