"""v2 placeholder for 03a_qubit_spectroscopy."""

from .pending import PendingCalibration


class QubitSpectroscopy(PendingCalibration):
    legacy_file = "03a_qubit_spectroscopy.py"

    def __init__(self, **kwargs):
        super().__init__(name="03a_qubit_spectroscopy", **kwargs)
