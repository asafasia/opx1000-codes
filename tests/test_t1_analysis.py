import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from calibration_utils.T1.analysis import FIT_VALUES, fit_raw_data
from calibration_utils.T1.plotting import plot_individual_data_with_fit


def make_fit(decay):
    values = np.zeros((1, len(FIT_VALUES)))
    values[0, FIT_VALUES.index("decay")] = decay
    values[0, FIT_VALUES.index("decay_decay")] = 1e-10
    return xr.DataArray(
        values,
        dims=["qubit", "fit_vals"],
        coords={"qubit": ["q1"], "fit_vals": FIT_VALUES},
    )


class T1AnalysisTests(unittest.TestCase):
    def test_failed_fit_is_marked_failed_without_raising(self):
        idle_time = np.arange(20.0)
        ds = xr.Dataset(
            {"state": (("qubit", "idle_time"), np.ones((1, len(idle_time))))},
            coords={"qubit": ["q1"], "idle_time": idle_time},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=True),
            log=lambda message: None,
        )

        with patch(
            "calibration_utils.T1.analysis.fit_decay_exp",
            side_effect=AttributeError("'NoneType' object has no attribute 'ndim'"),
        ):
            fit, results = fit_raw_data(ds, node)

        self.assertFalse(results["q1"].success)
        self.assertTrue(np.isnan(float(fit.sel(qubit="q1").tau.values)))

    def test_iq_readout_selects_valid_q_fit_when_i_fit_fails(self):
        idle_time = np.arange(20.0)
        decay = np.exp(-idle_time / 1000)
        ds = xr.Dataset(
            {
                "I": (("qubit", "idle_time"), np.ones((1, len(idle_time)))),
                "Q": (("qubit", "idle_time"), decay[None, :]),
            },
            coords={"qubit": ["q1"], "idle_time": idle_time},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(use_state_discrimination=False),
            log=lambda message: None,
        )

        with patch(
            "calibration_utils.T1.analysis.fit_decay_exp",
            side_effect=[AttributeError("failed I fit"), make_fit(-0.001)],
        ):
            fit, _ = fit_raw_data(ds, node)

        self.assertEqual(str(fit.sel(qubit="q1").selected_quadrature.values), "Q")

    def test_plot_uses_selected_q_quadrature(self):
        idle_time = np.arange(3.0)
        ds = xr.Dataset(
            {
                "I": (("qubit", "idle_time"), [[1.0, 1.0, 1.0]]),
                "Q": (("qubit", "idle_time"), [[3.0, 2.0, 1.0]]),
                "fit_data": (("qubit", "fit_vals"), make_fit(-0.001).values),
                "selected_quadrature": ("qubit", ["Q"]),
            },
            coords={"qubit": ["q1"], "idle_time": idle_time, "fit_vals": FIT_VALUES},
        ).assign_coords(tau=("qubit", [1000.0]), tau_error=("qubit", [10.0]), success=("qubit", [True]))
        fig, ax = plt.subplots()

        plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, ds.sel(qubit="q1"))

        self.assertEqual(ax.get_ylabel(), "Trans. amp. Q [mV]")
        self.assertEqual(ax.lines[0].get_marker(), ".")
        self.assertEqual(ax.lines[0].get_linestyle(), "-")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
