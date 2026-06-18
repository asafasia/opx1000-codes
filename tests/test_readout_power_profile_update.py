import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent


class ReadoutPowerProfileUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            REPOSITORY_ROOT / "calibrations_v2" / "08b_readout_power_optimization.py"
        ).read_text()

    def test_profile_update_stages_optimized_readout_amplitude(self):
        self.assertIn("pulses.json.pulses.{q.name}.readout.amplitude", self.source)
        self.assertIn('fit_result["optimal_amplitude"]', self.source)
        self.assertIn("ProfileUpdater().stage", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)

    def test_profile_update_keeps_reset_specific_fidelity_metric(self):
        self.assertIn('node.parameters.reset_type in {"active", "thermal"}', self.source)
        self.assertIn(
            "metrics.json.qubits.{q.name}.readout.fidelity_percent.{node.parameters.reset_type}",
            self.source,
        )
        self.assertIn('fit_result["readout_fidelity"]', self.source)


if __name__ == "__main__":
    unittest.main()
