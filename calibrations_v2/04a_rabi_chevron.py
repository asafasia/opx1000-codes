"""v2 placeholder for 04a_rabi_chevron."""

from .pending import PendingCalibration


class RabiChevron(PendingCalibration):
    legacy_file = "04a_rabi_chevron.py"

    def __init__(self, **kwargs):
        super().__init__(name="04a_rabi_chevron", **kwargs)
