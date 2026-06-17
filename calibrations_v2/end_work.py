"""v2 placeholder for end_work."""

from .pending import PendingCalibration


class EndWork(PendingCalibration):
    legacy_file = "end_work.py"

    def __init__(self, **kwargs):
        super().__init__(name="end_work", **kwargs)
