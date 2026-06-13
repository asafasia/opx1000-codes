import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parent.parent


class ReadoutFrequencyUpdaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            REPOSITORY_ROOT / "calibrations" / "08a_readout_frequency_optimization.py"
        ).read_text(encoding="utf-8")

    def test_successful_frequency_is_staged_through_profile_updater(self):
        self.assertIn("from updater import ProfileUpdater", self.source)
        self.assertIn(
            'f"qubits.json.qubits.{q.name}.frequencies_hz.resonator"',
            self.source,
        )
        self.assertIn("ProfileUpdater().stage(", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)

    def test_frequency_is_not_persisted_by_direct_state_save(self):
        update_block = self.source.split("def update_state", 1)[1].split(
            "# %% {Propose_profile_update}", 1
        )[0]
        self.assertNotIn("q.resonator.f_01", update_block)
        self.assertNotIn("q.resonator.RF_frequency", update_block)
        self.assertNotIn("node.save()", self.source)


if __name__ == "__main__":
    unittest.main()
