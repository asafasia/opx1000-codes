"""v2 placeholder for 04d_power_rabi_chevron."""

from .pending import PendingCalibration


class PowerRabiChevron(PendingCalibration):
    legacy_file = "04d_power_rabi_chevron.py"

    def __init__(self, **kwargs):
        super().__init__(name="04d_power_rabi_chevron", **kwargs)
