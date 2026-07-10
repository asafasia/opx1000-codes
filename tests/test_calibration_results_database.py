import tempfile
import unittest
from pathlib import Path

from calibrations_v2.results_db import CalibrationResultsDatabase


class CalibrationResultsDatabaseTests(unittest.TestCase):
    def test_records_runs_metrics_and_profile_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            database = CalibrationResultsDatabase(Path(directory) / "results.sqlite")
            run_id = database.record_run(
                calibration_name="t1",
                status="completed",
                mode="execute",
                profile_name="main",
                selected_qubit="q3",
                parameters={"num_averages": 1000},
                outcomes={"q3": "successful"},
            )
            database.record_metric(
                run_id, target_name="q3", metric_name="t1_ns", value=12345.0,
                uncertainty=120.0, unit="ns", accepted=True,
            )
            database.record_profile_update(
                run_id, field_path="qubits.json.qubits.q3.frequencies_hz.xy",
                previous_value=5_000_000_000, proposed_value=5_001_000_000,
            )

            history = database.latest_metrics("q3", "t1_ns")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["value"], 12345.0)
        self.assertEqual(history[0]["unit"], "ns")
        self.assertTrue(history[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
