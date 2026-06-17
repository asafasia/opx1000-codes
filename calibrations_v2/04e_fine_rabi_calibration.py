"""v2 placeholder for 04e_fine_rabi_calibration."""

from .pending import PendingCalibration


class FineRabiCalibration(PendingCalibration):
    legacy_file = "04e_fine_rabi_calibration.py"

    def __init__(self, **kwargs):
        super().__init__(name="04e_fine_rabi_calibration", **kwargs)
