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
        self.assertIn("node.parameters.xy_to_readout_delay_in_ns * u.ns", self.source)
        self.assertIn(
            "with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:",
            self.source,
        )

    def test_simulation_uses_short_representative_programs(self):
        self.assertIn("initialization_wait_in_ns: int = 200_000", self.source)
        self.assertIn("initialization_wait_in_ns=100", self.source)
        self.assertIn('job.wait_until("Done"', self.source)
        self.assertIn("except QMSimulationError:", self.source)
        self.assertIn("wf_report.create_plot(samples=None", self.source)
        self.assertIn('simulations[state] = {"figure": fig}', self.source)

    def test_merges_into_current_analysis_variable_names(self):
        self.assertIn('dataset.rename({"I": f"I{suffix}", "Q": f"Q{suffix}"})', self.source)
        self.assertIn("process_raw_dataset(xr.merge([ground, excited]), node)", self.source)
        self.assertIn("fit_raw_data(node.results[\"ds_raw\"], node)", self.source)
        self.assertIn("plot_iq_blobs_dashboard(", self.source)

    def test_successful_fit_updates_profile_angle_and_threshold(self):
        self.assertIn("record_state_updates", self.source)
        self.assertIn("readout.integration_weights_angle_rad", self.source)
        self.assertIn("readout.threshold", self.source)
        self.assertIn("proposing fitted parameters despite failed IQ-blob quality checks", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)


if __name__ == "__main__":
    unittest.main()
