"""v2 placeholder for 07_iq_blobs."""

from .pending import PendingCalibration


class IqBlobs(PendingCalibration):
    legacy_file = "07_iq_blobs.py"

    def __init__(self, **kwargs):
        super().__init__(name="07_iq_blobs", **kwargs)
