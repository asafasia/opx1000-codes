import unittest
from pathlib import Path


class PowerRabiEFSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "04b_power_rabi.py"
        ).read_text()

    def test_combined_node_selects_ef_transition_by_parameter(self):
        self.assertIn('return "EF_x180" if parameters.transition == "ef" else parameters.operation', self.source)
        self.assertIn('if self.parameters.transition == "ef":', self.source)

    def test_ef_transition_repeats_ef_pulse_with_npi(self):
        ef_block = self.source.split('if self.parameters.transition == "ef":', 1)[1].split("else:", 1)[0]

        self.assertIn("with for_(count, 0, count < npi, count + 1):", ef_block)
        self.assertIn('qubit.xy.play("EF_x180", amplitude_scale=a)', ef_block)

    def test_plotting_uses_state_discrimination_parameter(self):
        self.assertIn("self._validate_readout_dataset(ds)", self.source)
        self.assertIn("self.parameters.use_state_discrimination", self.source)

    def test_readout_basis_is_explicit_for_either_drive_transition(self):
        self.assertIn("self.readout_state(qubit, state[i])", self.source)
        self.assertIn("self.reset_qubit(qubit)", self.source)
        self.assertNotIn("and has_gef_readout_calibration(qubit)", self.source)

    def test_saves_raw_data_and_figures_like_standard_power_rabi(self):
        self.assertIn("class PowerRabi(BaseCalibration", self.source)
        self.assertNotIn("def save_raw_results(", self.source)
        self.assertNotIn("def save_figures(", self.source)


if __name__ == "__main__":
    unittest.main()
