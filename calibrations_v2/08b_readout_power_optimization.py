"""v2 placeholder for 08b_readout_power_optimization."""

from .pending import PendingCalibration


class ReadoutPowerOptimization(PendingCalibration):
    legacy_file = "08b_readout_power_optimization.py"

    def __init__(self, **kwargs):
        super().__init__(name="08b_readout_power_optimization", **kwargs)
