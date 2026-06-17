"""v2 placeholder for 14_gef_readout_frequency_optimization."""

from .pending import PendingCalibration


class GefReadoutFrequencyOptimization(PendingCalibration):
    legacy_file = "14_gef_readout_frequency_optimization.py"

    def __init__(self, **kwargs):
        super().__init__(name="14_gef_readout_frequency_optimization", **kwargs)
