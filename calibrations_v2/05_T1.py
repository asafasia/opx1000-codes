"""v2 placeholder for 05_T1."""

from .pending import PendingCalibration


class T1(PendingCalibration):
    legacy_file = "05_T1.py"

    def __init__(self, **kwargs):
        super().__init__(name="05_T1", **kwargs)
