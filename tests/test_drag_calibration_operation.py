import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

from calibration_utils.drag_calibration_180_minus180.analysis import process_raw_dataset
from calibration_utils.drag_calibration_180_minus180.parameters import Parameters


class DragCalibrationOperationTests(unittest.TestCase):
    def test_defaults_to_drag_operation_with_nonzero_alpha_setpoint(self):
        parameters = Parameters()

        self.assertEqual(parameters.operation, "x180_drag")
        self.assertNotEqual(parameters.alpha_setpoint, 0)

    def test_square_pulse_fails_with_clear_analysis_error(self):
        ds = xr.Dataset(
            {"state": (("qubit", "alpha_prefactor"), [[0.0, 1.0]])},
            coords={"qubit": ["q1"], "alpha_prefactor": [-1.0, 1.0]},
        )
        node = SimpleNamespace(
            parameters=SimpleNamespace(operation="x180", use_state_discrimination=True),
            namespace={
                "qubits": [
                    SimpleNamespace(
                        name="q1",
                        xy=SimpleNamespace(operations={"x180": SimpleNamespace()}),
                    )
                ]
            },
        )

        with self.assertRaisesRegex(ValueError, "not a DRAG pulse"):
            process_raw_dataset(ds, node)

    def test_sequence_accepts_named_x180_drag_operation(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "10b_drag_calibration_180_minus_180.py"
        ).read_text()

        self.assertIn('node.parameters.operation.startswith("x180")', source)
        self.assertIn("cannot be DRAG-calibrated", source)

    def test_sequence_uses_robust_simulation_helper(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "10b_drag_calibration_180_minus_180.py"
        ).read_text()

        self.assertIn("from utils.simulation import simulate_and_plot", source)
        self.assertNotIn("from qualibration_libs.runtime import simulate_and_plot", source)

    def test_sequence_does_not_save_incompatible_waveform_report_during_simulation(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "10b_drag_calibration_180_minus_180.py"
        ).read_text()

        self.assertIn(
            "@node.run_action(skip_if=node.parameters.simulate)\ndef save_results",
            source,
        )

    def test_sequence_proposes_drag_beta_profile_update(self):
        source = (
            Path(__file__).parent.parent
            / "calibrations"
            / "10b_drag_calibration_180_minus_180.py"
        ).read_text()

        self.assertIn("from updater import ProfileUpdater", source)
        self.assertIn('updates[f"pulses.json.pulses.{q.name}.{pulse_name}.beta"]', source)
        self.assertIn("ProfileUpdater().stage(", source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", source)


if __name__ == "__main__":
    unittest.main()
