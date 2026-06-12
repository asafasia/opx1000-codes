import unittest
from pathlib import Path


class SeparateIQBlobsSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "07_iq_blobs_separate.py"
        ).read_text()

    def test_builds_two_independent_state_programs(self):
        self.assertIn("node.machine = create_machine()", self.source)
        self.assertNotIn("node.machine.qmm.close_all_qms()", self.source)
        self.assertIn("def make_state_program", self.source)
        self.assertIn('state: make_state_program(node, state) for state in ("g", "e")', self.source)
        self.assertIn('ground = acquire_state(node, qmm, config, "g")', self.source)
        self.assertIn('excited = acquire_state(node, qmm, config, "e")', self.source)
        self.assertIn("node.parameters.qubit_operation", self.source)
        self.assertIn("node.parameters.qubit_amplitude_factor", self.source)
        self.assertIn("node.parameters.pi_repetitions", self.source)
        self.assertIn(
            "with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:",
            self.source,
        )

    def test_simulation_uses_short_representative_programs(self):
        self.assertIn("make_state_program(node, state, n_runs=1)", self.source)

    def test_merges_into_current_analysis_variable_names(self):
        self.assertIn('dataset.rename({"I": f"I{suffix}", "Q": f"Q{suffix}"})', self.source)
        self.assertIn("process_raw_dataset(xr.merge([ground, excited]), node)", self.source)
        self.assertIn("fit_raw_data(node.results[\"ds_raw\"], node)", self.source)
        self.assertIn("plot_iq_blobs_dashboard(", self.source)

    def test_diagnostic_copy_does_not_update_state(self):
        self.assertNotIn("record_state_updates", self.source)


if __name__ == "__main__":
    unittest.main()
