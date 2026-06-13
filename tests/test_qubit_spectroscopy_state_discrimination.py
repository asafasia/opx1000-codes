import unittest
from pathlib import Path


class QubitSpectroscopyStateDiscriminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "03a_qubit_spectroscopy.py"
        ).read_text()

    def test_state_mode_acquires_only_state_stream(self):
        self.assertIn("if node.parameters.use_state_discrimination:", self.source)
        self.assertIn("qubit.readout_state(state[i])", self.source)
        self.assertIn('save(f"state{i + 1}")', self.source)
        self.assertIn("validate_readout_dataset(dataset, node.parameters.use_state_discrimination)", self.source)

    def test_plot_receives_state_discrimination_parameter(self):
        self.assertIn(
            "use_state_discrimination=node.parameters.use_state_discrimination",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
