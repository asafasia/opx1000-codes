import unittest
from pathlib import Path


class MinimalIQBlobsSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parent.parent / "calibrations" / "07_iq_blobs_minimal.py"
        ).read_text(encoding="utf-8")

    def test_is_fixed_frequency_raw_acquisition_only(self):
        self.assertNotIn("update_frequency", self.source)
        self.assertNotIn("fit_raw_data", self.source)
        self.assertNotIn("analyse_data", self.source)
        self.assertNotIn("record_state_updates", self.source)

    def test_preserves_expected_raw_variables_without_averaging(self):
        self.assertIn('Ig_st[i].buffer(n_runs).save(f"Ig{i + 1}")', self.source)
        self.assertIn('Qg_st[i].buffer(n_runs).save(f"Qg{i + 1}")', self.source)
        self.assertIn('Im_st[i].buffer(n_runs).save(f"Im{i + 1}")', self.source)
        self.assertIn('Qm_st[i].buffer(n_runs).save(f"Qm{i + 1}")', self.source)
        self.assertNotIn(".average()", self.source)
        self.assertIn('node.results["ds_raw"][["Ig", "Qg", "Im", "Qm"]]', self.source)

    def test_plot_is_separate_and_non_blocking(self):
        save_block = self.source.split("def save_raw_results", 1)[1].split(
            "# %% {Plot_raw_results}", 1
        )[0]
        self.assertNotIn("plt.", save_block)
        self.assertIn("def plot_raw_results", self.source)
        self.assertIn("axis.scatter(selected.Ig, selected.Qg", self.source)
        self.assertIn("axis.scatter(selected.Im, selected.Qm", self.source)
        self.assertIn("float(selected.Ig.mean())", self.source)
        self.assertIn("float(selected.Qg.mean())", self.source)
        self.assertIn("float(selected.Im.mean())", self.source)
        self.assertIn("float(selected.Qm.mean())", self.source)
        self.assertIn("plt.show(block=False)", self.source)


if __name__ == "__main__":
    unittest.main()
