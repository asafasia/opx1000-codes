"""v2 placeholder for 02a_resonator_spectroscopy."""

from .pending import PendingCalibration


class ResonatorSpectroscopy(PendingCalibration):
    legacy_file = "02a_resonator_spectroscopy.py"

    def __init__(self, **kwargs):
        super().__init__(name="02a_resonator_spectroscopy", **kwargs)
