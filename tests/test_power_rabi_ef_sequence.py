import unittest
from pathlib import Path


class PowerRabiEFSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "13_power_rabi_ef.py"
        ).read_text()

    def test_plotting_uses_state_discrimination_parameter(self):
        self.assertIn("validate_readout_dataset(dataset, node.parameters.use_state_discrimination)", self.source)
        self.assertIn("use_state_discrimination=node.parameters.use_state_discrimination", self.source)

    def test_missing_optional_readout_shifts_do_not_break_program_creation(self):
        self.assertNotIn("qubit.resonator.intermediate_frequency\n                    +", self.source)
        self.assertIn("if has_gef_readout_calibration(qubit):", self.source)
        self.assertIn("qubit.readout_state(state[i])", self.source)

    def test_saves_raw_data_and_figures_like_standard_power_rabi(self):
        self.assertIn("CalibrationSaver().save_xarray(", self.source)
        self.assertIn("CalibrationSaver().save_figures(", self.source)

    def test_uses_confirmed_profile_updater_for_ef_pulse(self):
        self.assertIn('operations"]["EF_x180"]', self.source)
        self.assertIn("ProfileUpdater().stage(", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)


if __name__ == "__main__":
    unittest.main()
