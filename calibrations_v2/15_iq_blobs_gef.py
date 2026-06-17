"""v2 placeholder for 15_iq_blobs_gef."""

from .pending import PendingCalibration


class IqBlobsGef(PendingCalibration):
    legacy_file = "15_iq_blobs_gef.py"

    def __init__(self, **kwargs):
        super().__init__(name="15_iq_blobs_gef", **kwargs)
