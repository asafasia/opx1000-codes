"""v2 placeholder for 07_iq_blobs_separate."""

from .pending import PendingCalibration


class IqBlobsSeparate(PendingCalibration):
    legacy_file = "07_iq_blobs_separate.py"

    def __init__(self, **kwargs):
        super().__init__(name="07_iq_blobs_separate", **kwargs)
