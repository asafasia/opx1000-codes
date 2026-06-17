import unittest
from pathlib import Path


class PowerRabiEFSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "04b_power_rabi.py"
        ).read_text()

    def test_combined_node_selects_ef_transition_by_parameter(self):
        self.assertIn('node.parameters.transition = "ef"', self.source)
        self.assertIn('return "EF_x180" if parameters.transition == "ef" else parameters.operation', self.source)
        self.assertIn('if node.parameters.transition == "ef":', self.source)

    def test_ef_transition_repeats_ef_pulse_with_npi(self):
        ef_block = self.source.split('if node.parameters.transition == "ef":', 1)[1].split("else:", 1)[0]

        self.assertIn("with for_(count, 0, count < npi, count + 1):", ef_block)
        self.assertIn('qubit.xy.play("EF_x180", amplitude_scale=a)', ef_block)

    def test_plotting_uses_state_discrimination_parameter(self):
        self.assertIn("validate_readout_dataset(dataset, node.parameters.use_state_discrimination)", self.source)
        self.assertIn("use_state_discrimination=node.parameters.use_state_discrimination", self.source)

    def test_missing_optional_readout_shifts_do_not_break_program_creation(self):
        self.assertNotIn("qubit.resonator.intermediate_frequency\n                    +", self.source)
        self.assertIn("if has_gef_readout_calibration(qubit):", self.source)
        self.assertIn('getattr(qubit.resonator, "GEF_frequency_shift", None) is not None', self.source)
        self.assertIn("qubit.readout_state(state[i])", self.source)

    def test_saves_raw_data_and_figures_like_standard_power_rabi(self):
        self.assertIn("CalibrationSaver().save_xarray(", self.source)
        self.assertIn("CalibrationSaver().save_figures(", self.source)

    def test_uses_confirmed_profile_updater_for_ef_pulse(self):
        self.assertIn("operation = active_operation(node.parameters)", self.source)
        self.assertIn("ProfileUpdater().stage(", self.source)
        self.assertIn("ProfileUpdater().confirm_and_apply(proposal)", self.source)


if __name__ == "__main__":
    unittest.main()
