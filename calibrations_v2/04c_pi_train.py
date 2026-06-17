"""v2 placeholder for 04c_pi_train."""

from .pending import PendingCalibration


class PiTrain(PendingCalibration):
    legacy_file = "04c_pi_train.py"

    def __init__(self, **kwargs):
        super().__init__(name="04c_pi_train", **kwargs)
