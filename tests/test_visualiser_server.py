import tempfile
import unittest
import io
import json
import zipfile
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

    def test_experiment_summary_prefers_figure_qubit_over_stale_single_qubit_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "07_iq_blobs" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            (run / "figures").mkdir()
            (run / "profile" / "profile.json").write_text(
                '{"name": "single_qubit", "active_qubits": ["q10"]}\n',
                encoding="utf-8",
            )
            (run / "figures" / "q2_iq_blobs.png").write_bytes(b"not really a png")
            np.savez_compressed(run / "results.npz", I=np.array([0.1]))

            with patch.object(server, "PROJECT_ROOT", root):
                summary = server.experiment_summary(run, "calibration", "07_iq_blobs")

            self.assertEqual(summary["qubits"], ["q2"])
            self.assertFalse(summary["has_data"])
            self.assertTrue(summary["has_figures"])
            self.assertFalse(summary["has_parameters"])

    def test_experiment_summary_reports_data_parameters_and_profile_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "07_iq_blobs" / "12-00-00-000000"
            update = root / "data" / "calibration_updates" / "2026-06-18" / "07_iq_blobs" / "12-00-00-111111"
            run.mkdir(parents=True)
            update.mkdir(parents=True)
            np.savez_compressed(run / "sweep.npz", qubit=np.array(["q3"]))
            np.savez_compressed(run / "results.npz", I=np.array([0.1]))
            (run / "parameters.json").write_text("{}\n", encoding="utf-8")

            with patch.object(server, "PROJECT_ROOT", root), patch.object(
                server, "CALIBRATION_UPDATE_ROOT", root / "data" / "calibration_updates"
            ):
                summary = server.experiment_summary(run, "calibration", "07_iq_blobs")

            self.assertEqual(summary["qubits"], ["q3"])
            self.assertTrue(summary["has_data"])
            self.assertTrue(summary["has_parameters"])
            self.assertTrue(summary["has_profile_update"])

    def test_experiment_detail_omits_profile_integration_weights_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            (run / "profile" / "pulses.json").write_text(
                json.dumps(
                    {
                        "pulses": {
                            "q1": {
                                "readout": {
                                    "length_ns": 1000,
                                    "integration_weights": [[1.0, 1000]],
                                }
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(server, "PROJECT_ROOT", root):
                detail = server.experiment_detail(run.relative_to(root).as_posix())

            pulses = next(item for item in detail["metadata"] if item["relative"] == "profile/pulses.json")
            readout = pulses["value"]["pulses"]["q1"]["readout"]
            self.assertEqual(readout["length_ns"], 1000)
            self.assertNotIn("integration_weights", readout)

    def test_experiment_detail_hides_profile_kernel_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            kernel_dir = run / "profile" / "kernels" / "10d_readout_weights_optimization" / "2026-06-19_19-54-51-978082"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "q1_readout_kernel.npz").write_bytes(b"junk")
            (run / "results.npz").write_bytes(b"real run artifact")

            with patch.object(server, "PROJECT_ROOT", root):
                detail = server.experiment_detail(run.relative_to(root).as_posix())

            artifact_paths = {item["relative"] for item in detail["artifacts"]}
            self.assertIn("results.npz", artifact_paths)
            self.assertNotIn(
                "profile/kernels/10d_readout_weights_optimization/2026-06-19_19-54-51-978082/q1_readout_kernel.npz",
                artifact_paths,
            )

    def test_download_zip_contains_clean_run_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile" / "kernels").mkdir(parents=True)
            np.savez_compressed(run / "sweep.npz", detuning=np.array([1, 2]))
            np.savez_compressed(run / "results.npz", I=np.array([0.1, 0.2]))
            (run / "metadata.json").write_text('{"experiment_name": "04b_power_rabi"}\n', encoding="utf-8")
            (run / "profile" / "pulses.json").write_text(
                '{"readout": {"length_ns": 1000, "integration_weights": [[1.0, 1000]]}}\n',
                encoding="utf-8",
            )
            (run / "profile" / "kernels" / "q1_readout_kernel.npz").write_bytes(b"junk")

            with patch.object(server, "PROJECT_ROOT", root):
                filename, body = server.download_zip(run.relative_to(root).as_posix())

            self.assertEqual(filename, "04b_power_rabi_12-00-00-000000_data.zip")
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"sweep.npz", "results.npz", "metadata.json", "profile/pulses.json"},
                )
                profile = json.loads(archive.read("profile/pulses.json").decode("utf-8"))
            self.assertNotIn("integration_weights", profile["readout"])

    def test_download_npz_bundle_loads_in_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            np.savez_compressed(run / "sweep.npz", detuning=np.array([1, 2]))
            np.savez_compressed(run / "results.npz", I=np.array([0.1, 0.2]))
            (run / "profile" / "profile.json").write_text('{"name": "main"}\n', encoding="utf-8")

            with patch.object(server, "PROJECT_ROOT", root):
                filename, body = server.download_npz_bundle(run.relative_to(root).as_posix())

            self.assertEqual(filename, "04b_power_rabi_12-00-00-000000_data_bundle.npz")
            with np.load(io.BytesIO(body), allow_pickle=False) as bundle:
                np.testing.assert_array_equal(bundle["data__detuning"], [1, 2])
                np.testing.assert_allclose(bundle["data__I"], [0.1, 0.2])
                self.assertEqual(json.loads(str(bundle["profile__profile.json"]))["name"], "main")

    def test_download_json_bundle_loads_as_python_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            (run / "profile").mkdir(parents=True)
            np.savez_compressed(run / "sweep.npz", detuning=np.array([1, 2]))
            np.savez_compressed(run / "results.npz", I=np.array([0.1, 0.2]))
            (run / "metadata.json").write_text('{"experiment_name": "04b_power_rabi"}\n', encoding="utf-8")
            (run / "parameters.json").write_text('{"num_shots": 2}\n', encoding="utf-8")
            (run / "profile" / "pulses.json").write_text(
                '{"readout": {"length_ns": 1000, "integration_weights": [[1.0, 1000]]}}\n',
                encoding="utf-8",
            )

            with patch.object(server, "PROJECT_ROOT", root):
                filename, body = server.download_json_bundle(run.relative_to(root).as_posix())

            self.assertEqual(filename, "04b_power_rabi_12-00-00-000000_data.json")
            bundle = json.loads(body.decode("utf-8"))
            self.assertEqual(bundle["data"]["detuning"], [1, 2])
            self.assertEqual(bundle["data"]["I"], [0.1, 0.2])
            self.assertEqual(bundle["metadata"]["experiment_name"], "04b_power_rabi")
            self.assertEqual(bundle["parameters"]["num_shots"], 2)
            self.assertNotIn("integration_weights", bundle["profile"]["pulses.json"]["readout"])

    def test_open_result_folder_launches_resolved_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "calibrations" / "2026-06-18" / "04b_power_rabi" / "12-00-00-000000"
            run.mkdir(parents=True)

            with patch.object(server, "PROJECT_ROOT", root), patch.object(
                server.os, "name", "nt"
            ), patch.object(server.os, "startfile", create=True) as startfile:
                result = server.open_result_folder(run.relative_to(root).as_posix())

            self.assertEqual(
                result["opened"],
                "data/calibrations/2026-06-18/04b_power_rabi/12-00-00-000000",
            )
            startfile.assert_called_once_with(run)

    def test_parameter_scan_detail_links_scan_manifest_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "data" / "parameter_scans" / "2026-06-28" / "codex_t1_q1_check" / "21-13-51-592429"
            script = root / "calibrations_v2" / "05_T1.py"
            run.mkdir(parents=True)
            script.parent.mkdir(parents=True)
            script.write_text("# fake T1 calibration\n", encoding="utf-8")
            (run / "summary.csv").write_text(
                "timestamp,cycle,experiment_name,script,status,qubit,parameter,value,unit,success,duration_s,error\n"
                "2026-06-28T21:13:51+03:00,1,05_T1,calibrations_v2/05_T1.py,ok,q1,T1,47125,ns,True,15,\n",
                encoding="utf-8",
            )
            (run / "scan.json").write_text(
                json.dumps(
                    {
                        "name": "codex_t1_q1_check",
                        "experiments": [{"script": "calibrations_v2/05_T1.py"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(server, "PROJECT_ROOT", root):
                detail = server.experiment_detail(run.relative_to(root).as_posix())

            self.assertEqual(detail["summary"]["kind"], "parameter_scan")
            self.assertEqual(
                [item["project_path"] for item in detail["calibrations"]["scripts"]],
                ["calibrations_v2/05_T1.py"],
            )


if __name__ == "__main__":
    unittest.main()
