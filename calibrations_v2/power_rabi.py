"""Compatibility import for the renamed v2 Power Rabi module."""

from importlib import import_module

_module = import_module("calibrations_v2.04b_power_rabi")

PowerRabi = _module.PowerRabi
Parameters = _module.Parameters
active_operation = _module.active_operation
ensure_operation_available = _module.ensure_operation_available
has_gef_readout_calibration = _module.has_gef_readout_calibration
validate_readout_dataset = _module.validate_readout_dataset

__all__ = [
    "Parameters",
    "PowerRabi",
    "active_operation",
    "ensure_operation_available",
    "has_gef_readout_calibration",
    "validate_readout_dataset",
]
