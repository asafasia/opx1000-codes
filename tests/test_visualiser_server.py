import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from apps.visualiser import server


class VisualiserServerTests(unittest.TestCase):
    def test_experiment_summary_prefers_saved_sweep_qubits_over_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            (run / "profile" / "profile.json").write_text('{"active_qubits": ["q10"]}\n', encoding="utf-8")
            np.savez_compressed(run / "sweep.npz", qubit=np.array(["q3", "q3", "q4"]))

            with patch.object(server, "PROJECT_ROOT", root):
                summary = server.experiment_summary(run, "calibration", "04b_power_rabi")

            self.assertEqual(summary["qubits"], ["q3", "q4"])

    def test_experiment_summary_falls_back_to_manifest_active_qubits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            (run / "profile" / "profile.json").write_text(
                '{"manifest": {"active_qubits": ["q7"]}}\n',
                encoding="utf-8",
            )

            with patch.object(server, "PROJECT_ROOT", root):
                summary = server.experiment_summary(run, "calibration", "04b_power_rabi")

            self.assertEqual(summary["qubits"], ["q7"])


if __name__ == "__main__":
    unittest.main()
