"""v2 placeholder for 00_hello_qua."""

from .pending import PendingCalibration


class HelloQua(PendingCalibration):
    legacy_file = "00_hello_qua.py"

    def __init__(self, **kwargs):
        super().__init__(name="00_hello_qua", **kwargs)
