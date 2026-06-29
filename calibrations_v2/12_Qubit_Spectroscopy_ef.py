"""Compatibility wrapper for the merged EF qubit spectroscopy calibration."""

from __future__ import annotations

import sys
from pathlib import Path
from importlib import import_module

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from quam_config import create_machine

if __package__ in {None, ""}:
    from calibrations_v2.core import CalibrationOptions
else:
    from .core import CalibrationOptions

_qubit_spectroscopy = import_module("calibrations_v2.03a_qubit_spectroscopy")
Parameters = _qubit_spectroscopy.Parameters
QubitSpectroscopy = _qubit_spectroscopy.QubitSpectroscopy


class QubitSpectroscopyEf(QubitSpectroscopy):
    """Backward-compatible alias using ``transition='ef'``."""

    def __init__(self, parameters: Parameters, *args, **kwargs) -> None:
        parameters.transition = "ef"
        super().__init__(parameters, *args, **kwargs)
        self.name = "12_Qubit_Spectroscopy_ef"


if __name__ == "__main__":
    parameters = Parameters()
    parameters.transition = "ef"
    parameters.num_shots = 100
    parameters.frequency_span_in_mhz = 200
    parameters.frequency_step_in_mhz = 2
    parameters.operation_amplitude_factor = 0.9

    calibration = QubitSpectroscopy(
        parameters=parameters,
        options=CalibrationOptions(),
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
