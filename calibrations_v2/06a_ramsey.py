"""v2 placeholder for 06a_ramsey."""

from .pending import PendingCalibration


class Ramsey(PendingCalibration):
    legacy_file = "06a_ramsey.py"

    def __init__(self, **kwargs):
        super().__init__(name="06a_ramsey", **kwargs)
