import unittest
from datetime import datetime, timedelta, timezone

from apps.parameter_scan.server import analyze_points, available_qubits, experiment_scripts, group_series


class ParameterScanAppTests(unittest.TestCase):
    def test_analyze_points_reports_variation_and_drift_per_hour(self):
        start = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        points = [
            {"timestamp_epoch": (start + timedelta(hours=index)).timestamp(), "value": 10.0 + index}
            for index in range(4)
        ]

        analysis = analyze_points(points)

        self.assertEqual(analysis["count"], 4)
        self.assertAlmostEqual(analysis["drift_per_hour"], 1.0)
        self.assertGreater(analysis["variance"], 0)
        self.assertIn(analysis["verdict"], {"watch", "drifting"})

    def test_group_series_ignores_error_rows(self):
        rows = [
            {
                "status": "ok",
                "experiment_name": "05_T1",
                "qubit": "q1",
                "parameter": "T1",
                "unit": "ns",
                "value": 1000.0,
                "timestamp_epoch": 1.0,
                "timestamp": "t1",
                "cycle": "1",
                "success": "True",
            },
            {"status": "error", "experiment_name": "05_T1", "value": None},
        ]

        series = group_series(rows)

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["parameter"], "T1")
        self.assertEqual(series[0]["analysis"]["latest"], 1000.0)

    def test_available_qubits_reads_single_qubit_profile(self):
        qubits = available_qubits()

        self.assertIn("q1", qubits)
        self.assertIn("q10", qubits)
        self.assertLess(qubits.index("q2"), qubits.index("q10"))

    def test_experiment_scripts_are_limited_to_live_scan_set(self):
        scripts = experiment_scripts()

        self.assertEqual(
            [item["script"] for item in scripts],
            [
                "calibrations_v2/03a_qubit_spectroscopy.py",
                "calibrations_v2/05_T1.py",
                "calibrations_v2/06a_ramsey.py",
                "calibrations_v2/07_iq_blobs.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
