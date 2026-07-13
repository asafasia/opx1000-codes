"""Lazy registry for terminal-friendly calibration names."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class CalibrationEntry:
    """Import recipe for one calibration class."""

    key: str
    module: str
    class_name: str
    description: str = ""
    parameters_module: str | None = None

    def load_class(self) -> type[Any]:
        module = import_module(self.module)
        return getattr(module, self.class_name)

    def load_parameters_class(self) -> type[Any]:
        if self.parameters_module is not None:
            module = import_module(self.parameters_module)
            return getattr(module, "Parameters")
        calibration_class = self.load_class()
        module = import_module(calibration_class.__module__)
        return getattr(module, "Parameters")


CALIBRATIONS: dict[str, CalibrationEntry] = {
    "hello": CalibrationEntry("hello", "calibrations.00_hello_qua", "HelloQua"),
    "tof": CalibrationEntry(
        "tof",
        "calibrations.01b_time_of_flight_mw_fem",
        "TimeOfFlightMwFem",
        "Time of flight, readout offsets, and gains.",
        "calibration_utils.time_of_flight_mw.parameters",
    ),
    "resonator": CalibrationEntry(
        "resonator",
        "calibrations.02a_resonator_spectroscopy",
        "ResonatorSpectroscopy",
        "1D resonator spectroscopy.",
        "calibration_utils.resonator_spectroscopy.parameters",
    ),
    "resonator-power": CalibrationEntry(
        "resonator-power",
        "calibrations.02b_resonator_spectroscopy_vs_power",
        "ResonatorSpectroscopyVsPower",
        "Resonator spectroscopy versus readout power.",
        "calibration_utils.resonator_spectroscopy_vs_amplitude.parameters",
    ),
    "qubit": CalibrationEntry(
        "qubit",
        "calibrations.03a_qubit_spectroscopy",
        "QubitSpectroscopy",
        "Qubit spectroscopy.",
        "calibration_utils.qubit_spectroscopy.parameters",
    ),
    "rabi-chevron": CalibrationEntry(
        "rabi-chevron",
        "calibrations.04a_rabi_chevron",
        "RabiChevron",
        parameters_module="calibration_utils.rabi_chevron.parameters",
    ),
    "power-rabi": CalibrationEntry(
        "power-rabi",
        "calibrations.04b_power_rabi",
        "PowerRabi",
        "Power Rabi amplitude calibration.",
        "calibration_utils.power_rabi.parameters",
    ),
    "pi-train": CalibrationEntry(
        "pi-train",
        "calibrations.04c_pi_train",
        "PiTrain",
        parameters_module="calibration_utils.pi_train.parameters",
    ),
    "power-rabi-chevron": CalibrationEntry(
        "power-rabi-chevron",
        "calibrations.04d_power_rabi_chevron",
        "PowerRabiChevron",
        parameters_module="calibration_utils.power_rabi_chevron.parameters",
    ),
    "fine-rabi": CalibrationEntry(
        "fine-rabi",
        "calibrations.04e_fine_rabi_calibration",
        "FineRabiCalibration",
        parameters_module="calibration_utils.fine_rabi.parameters",
    ),
    "t1": CalibrationEntry(
        "t1",
        "calibrations.05_T1",
        "T1",
        "T1 relaxation.",
        "calibration_utils.T1.parameters",
    ),
    "ramsey": CalibrationEntry(
        "ramsey",
        "calibrations.06a_ramsey",
        "Ramsey",
        parameters_module="calibration_utils.ramsey.parameters",
    ),
    "echo": CalibrationEntry(
        "echo",
        "calibrations.06b_echo",
        "Echo",
        parameters_module="calibration_utils.T2echo.parameters",
    ),
    "iq-blobs": CalibrationEntry(
        "iq-blobs",
        "calibrations.07_iq_blobs",
        "IqBlobs",
        parameters_module="calibration_utils.iq_blobs.parameters",
    ),
    "readout-frequency": CalibrationEntry(
        "readout-frequency",
        "calibrations.08a_readout_frequency_optimization",
        "ReadoutFrequencyOptimization",
        parameters_module="calibration_utils.readout_frequency_optimization.parameters",
    ),
    "readout-power": CalibrationEntry(
        "readout-power",
        "calibrations.08b_readout_power_optimization",
        "ReadoutPowerOptimization",
        parameters_module="calibration_utils.readout_power_optimization.parameters",
    ),
    "readout-frequency-amplitude": CalibrationEntry(
        "readout-frequency-amplitude",
        "calibrations.08c_readout_frequency_amplitude_optimization",
        "ReadoutFrequencyAmplitudeOptimization",
        "2D readout optimization over frequency and amplitude.",
        "calibration_utils.readout_frequency_amplitude_optimization.parameters",
    ),
    "drag": CalibrationEntry(
        "drag",
        "calibrations.10b_drag_calibration_180_minus_180",
        "DragCalibration180Minus180",
        parameters_module="calibration_utils.drag_calibration_180_minus180.parameters",
    ),
    "readout-weights": CalibrationEntry(
        "readout-weights",
        "calibrations.10d_readout_weights_optimization",
        "ReadoutWeightsOptimization",
        parameters_module="calibration_utils.readout_weights_optimization.parameters",
    ),
    "rb": CalibrationEntry(
        "rb",
        "calibrations.11a_single_qubit_randomized_benchmarking",
        "SingleQubitRandomizedBenchmarking",
        parameters_module="calibration_utils.single_qubit_randomized_benchmarking.parameters",
    ),
    "rb-interleaved": CalibrationEntry(
        "rb-interleaved",
        "calibrations.11b_single_qubit_randomized_benchmarking_interleaved",
        "SingleQubitRandomizedBenchmarkingInterleaved",
        parameters_module="calibration_utils.single_qubit_randomized_benchmarking_interleaved.parameters",
    ),
    "qubit-ef": CalibrationEntry(
        "qubit-ef",
        "calibrations.12_Qubit_Spectroscopy_ef",
        "QubitSpectroscopyEf",
        parameters_module="calibration_utils.qubit_spectroscopy.parameters",
    ),
    "gef-readout-frequency": CalibrationEntry(
        "gef-readout-frequency",
        "calibrations.14_gef_readout_frequency_optimization",
        "GefReadoutFrequencyOptimization",
        parameters_module="calibration_utils.readout_gef_frequency_optimization.parameters",
    ),
    "iq-blobs-gef": CalibrationEntry(
        "iq-blobs-gef",
        "calibrations.15_iq_blobs_gef",
        "IqBlobsGef",
        parameters_module="calibration_utils.iq_blobs_ef.parameters",
    ),
    "xy8": CalibrationEntry(
        "xy8",
        "calibrations.16_xy8",
        "XY8",
        parameters_module="calibration_utils.xy8.parameters",
    ),
    "cpmg": CalibrationEntry(
        "cpmg",
        "calibrations.17_cpmg",
        "CPMG",
        parameters_module="calibration_utils.cpmg.parameters",
    ),
}


def get_entry(name: str) -> CalibrationEntry:
    """Resolve a friendly calibration name or exact module stem."""
    normalized = name.strip().replace("_", "-")
    if normalized in CALIBRATIONS:
        return CALIBRATIONS[normalized]
    for entry in CALIBRATIONS.values():
        module_stem = entry.module.rsplit(".", 1)[-1].replace("_", "-")
        if normalized == module_stem:
            return entry
    valid = ", ".join(sorted(CALIBRATIONS))
    raise KeyError(f"Unknown calibration {name!r}. Valid names: {valid}")
