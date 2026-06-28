import unittest
from dataclasses import dataclass

import numpy as np
import xarray as xr

from calibration_utils.analysis_base import AnalysisResult, BaseAnalysis


@dataclass
class Fit:
    value: np.float64
    success: bool


class ExampleAnalysis(BaseAnalysis):
    def process(self, ds):
        return ds.assign(processed=("x", np.array([2.0, 4.0])))

    def fit(self, ds):
        return ds.assign(fit=("x", ds.processed.values + 1.0)), {
            "q1": Fit(np.float64(3.0), True),
            "q2": {"value": np.float64(0.0), "success": False},
        }


class AnalysisBaseTests(unittest.TestCase):
    def test_analysis_result_to_dict_serializes_fit_results_and_dataset_summary(self):
        dataset = xr.Dataset(
            data_vars={"signal": ("x", np.array([1.0, 2.0]))},
            coords={"x": np.array([10, 20])},
        )
        result = AnalysisResult(
            ds_processed=dataset,
            ds_fit=dataset.assign(fit=("x", np.array([1.1, 2.1]))),
            fit_results={"q1": Fit(np.float64(23.0), True)},
            outcomes={"q1": "successful"},
            summary={"array": np.array([1, 2])},
        )

        serialized = result.to_dict()

        self.assertEqual(serialized["fit_results"]["q1"], {"value": 23.0, "success": True})
        self.assertEqual(serialized["outcomes"], {"q1": "successful"})
        self.assertEqual(serialized["summary"], {"array": [1, 2]})
        self.assertEqual(serialized["datasets"]["fit"]["dims"], {"x": 2})
        self.assertEqual(serialized["datasets"]["fit"]["data_vars"], ["fit", "signal"])

    def test_base_analysis_packages_plain_results_and_outcomes(self):
        dataset = xr.Dataset(
            data_vars={"signal": ("x", np.array([1.0, 2.0]))},
            coords={"x": np.array([10, 20])},
        )

        result = ExampleAnalysis(node=object()).run(dataset)

        self.assertIn("processed", result.ds_processed)
        self.assertIn("fit", result.ds_fit)
        self.assertEqual(result.fit_results["q1"], {"value": 3.0, "success": True})
        self.assertEqual(result.outcomes, {"q1": "successful", "q2": "failed"})
        self.assertEqual(result.summary["num_successful"], 1)


if __name__ == "__main__":
    unittest.main()
