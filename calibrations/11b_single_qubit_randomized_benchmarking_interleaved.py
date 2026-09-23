"""Compatibility entry point for matched-reference interleaved RB."""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from importlib import import_module
from calibration_utils.single_qubit_randomized_benchmarking_interleaved import Parameters

_BaseRB = import_module("calibrations.11a_single_qubit_randomized_benchmarking").SingleQubitRandomizedBenchmarking


class SingleQubitRandomizedBenchmarkingInterleaved(_BaseRB):
    def __init__(self, parameters: Parameters, machine=None, **kwargs):
        if parameters.mode != "interleaved":
            raise ValueError("The rb-interleaved entry point requires mode='interleaved'.")
        super().__init__(parameters=parameters, machine=machine, **kwargs)
        self.name = "11b_single_qubit_randomized_benchmarking_interleaved"


if __name__ == "__main__":
    from quam_config import create_machine
    from calibrations.core import CalibrationOptions

    parameters = Parameters(simulate=True)
    calibration = SingleQubitRandomizedBenchmarkingInterleaved(
        parameters, machine=create_machine(qubit="q1"),
        options=CalibrationOptions(save_raw_data=False, save_figures=False,
            update_state=False, propose_profile_update=False, apply_profile_update=False))
    calibration.run()
