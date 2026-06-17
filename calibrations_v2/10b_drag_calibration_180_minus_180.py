"""v2 placeholder for 10b_drag_calibration_180_minus_180."""

from .pending import PendingCalibration


class DragCalibration180Minus180(PendingCalibration):
    legacy_file = "10b_drag_calibration_180_minus_180.py"

    def __init__(self, **kwargs):
        super().__init__(name="10b_drag_calibration_180_minus_180", **kwargs)
