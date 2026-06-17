"""v2 placeholder for resonator_parameter_scan."""

from .pending import PendingCalibration


class ResonatorParameterScan(PendingCalibration):
    legacy_file = "resonator_parameter_scan.py"

    def __init__(self, **kwargs):
        super().__init__(name="resonator_parameter_scan", **kwargs)
