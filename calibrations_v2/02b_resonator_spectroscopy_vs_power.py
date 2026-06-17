"""v2 placeholder for 02b_resonator_spectroscopy_vs_power."""

from .pending import PendingCalibration


class ResonatorSpectroscopyVsPower(PendingCalibration):
    legacy_file = "02b_resonator_spectroscopy_vs_power.py"

    def __init__(self, **kwargs):
        super().__init__(name="02b_resonator_spectroscopy_vs_power", **kwargs)
