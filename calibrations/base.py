"""Backward-compatible imports for the calibration core API."""

import matplotlib.pyplot as plt

from .core import (
    BaseCalibration,
    CalibrationError,
    CalibrationOptions,
    CalibrationStatus,
)

__all__ = [
    "BaseCalibration",
    "CalibrationError",
    "CalibrationOptions",
    "CalibrationStatus",
]
