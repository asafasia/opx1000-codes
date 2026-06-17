"""v2 placeholder for 12_Qubit_Spectroscopy_ef."""

from .pending import PendingCalibration


class QubitSpectroscopyEf(PendingCalibration):
    legacy_file = "12_Qubit_Spectroscopy_ef.py"

    def __init__(self, **kwargs):
        super().__init__(name="12_Qubit_Spectroscopy_ef", **kwargs)
