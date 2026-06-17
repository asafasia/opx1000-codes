"""v2 placeholder for 11a_single_qubit_randomized_benchmarking."""

from .pending import PendingCalibration


class SingleQubitRandomizedBenchmarking(PendingCalibration):
    legacy_file = "11a_single_qubit_randomized_benchmarking.py"

    def __init__(self, **kwargs):
        super().__init__(name="11a_single_qubit_randomized_benchmarking", **kwargs)
