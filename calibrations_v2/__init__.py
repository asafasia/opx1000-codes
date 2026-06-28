"""Class-based calibration framework without QualibrationNode decorators."""

__all__ = [
    "BaseCalibration",
    "CalibrationError",
    "CalibrationOptions",
    "CalibrationStatus",
]


def __getattr__(name):
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(name)
