import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr

from calibrations.runtime_estimation import (
    estimate_runtime,
    format_duration,
    progress_counter,
    workload_units,
)


class RuntimeEstimationTests(unittest.TestCase):
    def test_workload_uses_non_qubit_sweep_axes_and_shots(self):
        axes = {
            "qubit": xr.DataArray(["q1", "q2"]),
            "frequency": xr.DataArray(np.arange(11)),
            "amplitude": xr.DataArray(np.arange(7)),
        }

        points, repetitions, workload = workload_units(
            axes, SimpleNamespace(num_shots=20), progress_total=20
        )

        self.assertEqual(points, 77)
        self.assertEqual(repetitions, 20)
        self.assertEqual(workload, 1540)

    def test_rb_workload_includes_sequences_and_shots(self):
        points, repetitions, workload = workload_units(
            {
                "nb_of_sequences": xr.DataArray(np.arange(30)),
                "depth": xr.DataArray(np.arange(5)),
            },
            SimpleNamespace(num_shots=10),
            progress_total=30,
        )

        self.assertEqual(points, 5)
        self.assertEqual(repetitions, 300)
        self.assertEqual(workload, 1500)

    def test_estimate_scales_a_comparable_saved_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = output_root / "2026-08-23" / "example" / "12-00-00-000000"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "example",
                        "execution_duration_s": 10.0,
                        "sweep": {
                            "qubit": {"shape": [1]},
                            "frequency": {"shape": [2]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "parameters.json").write_text(
                json.dumps({"num_shots": 5, "reset_type": "thermal"}),
                encoding="utf-8",
            )

            estimate = estimate_runtime(
                experiment_name="example",
                axes={"frequency": xr.DataArray(np.arange(4))},
                parameters=SimpleNamespace(num_shots=5, reset_type="thermal"),
                progress_total=5,
                output_root=output_root,
            )

        self.assertEqual(estimate.workload_units, 20)
        self.assertEqual(estimate.historical_runs, 1)
        self.assertAlmostEqual(estimate.estimated_seconds, 20.0)

    def test_incompatible_reset_history_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = output_root / "2026-08-23" / "example" / "12-00-00-000000"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "run_duration_s": 10.0,
                        "sweep": {"frequency": {"shape": [2]}},
                    }
                ),
                encoding="utf-8",
            )
            (run / "parameters.json").write_text(
                json.dumps({"num_shots": 5, "reset_type": "thermal"}),
                encoding="utf-8",
            )

            estimate = estimate_runtime(
                experiment_name="example",
                axes={"frequency": xr.DataArray(np.arange(4))},
                parameters=SimpleNamespace(num_shots=5, reset_type="active"),
                progress_total=5,
                output_root=output_root,
            )

        self.assertIsNone(estimate.estimated_seconds)
        self.assertEqual(estimate.historical_runs, 0)

    def test_progress_counter_reports_adaptive_eta(self):
        with patch("calibrations.runtime_estimation.time.time", return_value=130.0), patch(
            "builtins.print"
        ) as output:
            progress_counter(3, 10, start_time=100.0)

        message = output.call_args.args[0]
        self.assertIn("3/10 complete", message)
        self.assertIn("elapsed 30s", message)
        self.assertIn("ETA 1m 10s", message)

    def test_format_duration(self):
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m")


if __name__ == "__main__":
    unittest.main()
