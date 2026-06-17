"""v2 placeholder for 08a_readout_frequency_optimization."""

from .pending import PendingCalibration


class ReadoutFrequencyOptimization(PendingCalibration):
    legacy_file = "08a_readout_frequency_optimization.py"

    def __init__(self, **kwargs):
        super().__init__(name="08a_readout_frequency_optimization", **kwargs)
