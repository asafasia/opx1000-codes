import unittest
from types import SimpleNamespace
from unittest.mock import patch

from apps.job_status import server


class JobStatusAppTests(unittest.TestCase):
    @patch("apps.job_status.server.query_profile_qop_status")
    def test_jobs_payload_uses_shared_qop_status_query(self, query_mock):
        query_mock.return_value = SimpleNamespace(
            to_dict=lambda: {
                "open_qms": ["qm-1"],
                "has_active_jobs": True,
                "jobs": [{"id": "job-1", "status": "Running"}],
            }
        )

        payload = server.jobs_payload(
            profile_name="single_qubit",
            qubit="q3",
            all_jobs=True,
        )

        query_mock.assert_called_once_with(
            profile_name="single_qubit",
            qubit="q3",
            active_only=False,
        )
        self.assertEqual(payload["profile"], "single_qubit")
        self.assertEqual(payload["qubit"], "q3")
        self.assertFalse(payload["active_only"])
        self.assertTrue(payload["has_active_jobs"])
        self.assertEqual(payload["jobs"][0]["id"], "job-1")
        self.assertIn("polled_at", payload)

    @patch("apps.job_status.server.query_profile_qop_status")
    def test_jobs_payload_defaults_to_active_jobs(self, query_mock):
        query_mock.return_value = SimpleNamespace(
            to_dict=lambda: {
                "open_qms": [],
                "has_active_jobs": False,
                "jobs": [],
            }
        )

        payload = server.jobs_payload()

        query_mock.assert_called_once_with(
            profile_name=None,
            qubit=None,
            active_only=True,
        )
        self.assertEqual(payload["profile"], "default")
        self.assertIsNone(payload["qubit"])
        self.assertTrue(payload["active_only"])


if __name__ == "__main__":
    unittest.main()
