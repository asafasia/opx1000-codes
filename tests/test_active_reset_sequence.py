import unittest
from pathlib import Path


class ActiveResetSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent
            / "dynamic_circuit_active_reset"
            / "active_reset.py"
        ).read_text()

    def test_uses_qualibration_separate_acquisition_structure(self):
        self.assertIn("QualibrationNode[Parameters, Quam]", self.source)
        self.assertIn("def make_active_reset_program", self.source)
        self.assertIn('for preparation in ("g", "e")', self.source)
        self.assertIn('ground = acquire_preparation(node, qmm, config, "g")', self.source)
        self.assertIn('excited = acquire_preparation(node, qmm, config, "e")', self.source)
        self.assertIn("XarrayDataFetcher", self.source)
        self.assertIn("CalibrationSaver().save_xarray", self.source)

    def test_keeps_conditional_active_reset_and_before_after_measurements(self):
        self.assertIn("initial_i[i] > threshold", self.source)
        self.assertIn('qubit.xy.play("x180")', self.source)
        self.assertIn("qua_vars=(initial_i[i], initial_q[i])", self.source)
        self.assertIn("qua_vars=(final_i[i], final_q[i])", self.source)
        self.assertIn('save(f"initial_state{i + 1}")', self.source)
        self.assertIn('save(f"final_state{i + 1}")', self.source)
        self.assertIn('save(f"reset_applied{i + 1}")', self.source)

    def test_plots_before_and_after_reset(self):
        self.assertIn('axes[0].set_title("Before active reset")', self.source)
        self.assertIn('axes[1].set_title("After active reset")', self.source)
        self.assertIn('axes[2].set_title("Active-reset result")', self.source)


if __name__ == "__main__":
    unittest.main()
