import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).parent.parent
    / "Projects"
    / "shaped_pulse_spectroscopy"
    / "simulation"
    / "qutrit_slices.py"
)
SPEC = importlib.util.spec_from_file_location("qutrit_slices", MODULE_PATH)
qutrit_slices = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qutrit_slices
SPEC.loader.exec_module(qutrit_slices)


class QutritSliceSimulationTests(unittest.TestCase):
    def test_zero_drive_stays_in_ground_state(self):
        result = qutrit_slices.simulate_qutrit_slices(
            duration_us=20,
            detuning_mhz=np.array([-0.5, 0.0, 0.5]),
            rabi_mhz=np.array([0.0]),
            t1_us=12.1,
            t2_star_us=13.19,
            anharmonicity_mhz=-237.95,
            num_steps_per_half=10,
            cutoff=0.005,
            echo=True,
        )

        np.testing.assert_allclose(result.ground, 1.0)
        np.testing.assert_allclose(result.excited, 0.0)
        np.testing.assert_allclose(result.second_excited, 0.0)

    def test_echo_and_no_echo_return_physical_distinct_slices(self):
        common = {
            "duration_us": 1.0,
            "detuning_mhz": np.linspace(-0.2, 0.2, 5),
            "rabi_mhz": np.array([3.0]),
            "t1_us": 12.1,
            "t2_star_us": 13.19,
            "anharmonicity_mhz": -237.95,
            "num_steps_per_half": 1_000,
            "cutoff": 0.005,
        }
        no_echo = qutrit_slices.simulate_qutrit_slices(echo=False, **common)
        echo = qutrit_slices.simulate_qutrit_slices(echo=True, **common)

        for result in (no_echo, echo):
            total = result.ground + result.excited + result.second_excited
            np.testing.assert_allclose(total, 1.0, atol=1e-10)
            self.assertGreaterEqual(float(result.ground.min()), 0.0)
            self.assertLessEqual(float(result.ground.max()), 1.0)
        self.assertFalse(np.allclose(no_echo.excited, echo.excited))

    def test_beta_zero_with_transition_is_exactly_backward_compatible(self):
        common = {
            "duration_us": 1.0,
            "detuning_mhz": np.linspace(-0.2, 0.2, 3),
            "rabi_mhz": np.array([2.0, 4.0]),
            "t1_us": 48.82,
            "t2_star_us": 22.79,
            "anharmonicity_mhz": -237.95,
            "num_steps_per_half": 500,
            "cutoff": 0.005,
            "echo": True,
        }
        legacy = qutrit_slices.simulate_qutrit_slices(**common)
        explicit = qutrit_slices.simulate_qutrit_slices(
            drag_beta=0.0,
            echo_transition_time_ns=16.0,
            **common,
        )

        for name in ("ground", "excited", "second_excited"):
            np.testing.assert_array_equal(getattr(legacy, name), getattr(explicit, name))

    def test_nonzero_drag_with_smooth_echo_is_physical_and_changes_result(self):
        common = {
            "duration_us": 1.0,
            "detuning_mhz": np.linspace(-0.2, 0.2, 5),
            "rabi_mhz": np.array([3.0, 6.0]),
            "t1_us": 48.82,
            "t2_star_us": 22.79,
            "anharmonicity_mhz": -237.95,
            "num_steps_per_half": 1_000,
            "cutoff": 0.005,
            "echo": True,
        }
        baseline = qutrit_slices.simulate_qutrit_slices(**common)
        drag = qutrit_slices.simulate_qutrit_slices(
            drag_beta=-0.2,
            echo_transition_time_ns=16.0,
            **common,
        )

        total = drag.ground + drag.excited + drag.second_excited
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
        self.assertFalse(np.allclose(baseline.excited, drag.excited))


if __name__ == "__main__":
    unittest.main()
