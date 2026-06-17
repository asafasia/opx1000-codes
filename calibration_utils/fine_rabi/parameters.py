from typing import Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


RotationType = Literal["PI", "PI_HALF"]
AmpFactorSpacing = Literal["uniform", "center_dense"]


class NodeSpecificParameters(RunnableParameters):
    rotation_type: RotationType = "PI" 
    """Rotation to calibrate: PI uses x180 pairs, PI_HALF uses x90 quartets."""
    use_state_discrimination: bool = True
    """Measure discriminated state instead of raw I/Q. Default is True."""
    num_shots: int = 500
    """Number of averages for every amplitude and repetition point."""
    max_repetition_groups: int = 40
    """Maximum number of complete gate groups in the train."""
    min_amp_factor: float = 0.8
    """Minimum pulse amplitude scaling factor."""
    max_amp_factor: float = 1.2
    """Maximum pulse amplitude scaling factor, inclusive when on the step grid."""
    amp_factor_step: float = 0.01
    """Step size for the amplitude scaling factor."""
    amp_factor_spacing: AmpFactorSpacing = "uniform"
    """Amplitude scan spacing. Use "center_dense" to place more points around factor 1."""
    amp_factor_density_power: float = 2.0
    """Power-law clustering strength for center_dense spacing; larger values increase density near factor 1."""
    fourier_oversampling: int = 8
    """Zero-padding factor for the repetition-axis Fourier transform."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameters for the fine Rabi calibration experiment."""

    def get_amp_factors(self):
        if self.amp_factor_step <= 0:
            raise ValueError("amp_factor_step must be positive.")
        if self.max_amp_factor < self.min_amp_factor:
            raise ValueError("max_amp_factor must be greater than or equal to min_amp_factor.")
        count = int(np.floor((self.max_amp_factor - self.min_amp_factor) / self.amp_factor_step)) + 1
        if self.amp_factor_spacing == "uniform":
            amps = self.min_amp_factor + self.amp_factor_step * np.arange(count)
            if amps.size == 0 or not np.isclose(amps[-1], self.max_amp_factor):
                amps = np.append(amps, self.max_amp_factor)
            return amps.astype(float)
        if self.amp_factor_spacing == "center_dense":
            if self.amp_factor_density_power <= 1:
                raise ValueError("amp_factor_density_power must be greater than 1.")
            return _center_dense_amp_factors(
                self.min_amp_factor,
                self.max_amp_factor,
                count,
                self.amp_factor_step,
                self.amp_factor_density_power,
            )
        raise ValueError(f"Unsupported amp_factor_spacing {self.amp_factor_spacing!r}.")

    def get_repetition_groups(self):
        if self.max_repetition_groups < 0:
            raise ValueError("max_repetition_groups must be non-negative.")
        return np.arange(self.max_repetition_groups + 1, dtype=int)


def operation_for_rotation(rotation_type: RotationType) -> str:
    if rotation_type == "PI":
        return "x180"
    if rotation_type == "PI_HALF":
        return "x90"
    raise ValueError(f"Unsupported rotation_type {rotation_type!r}.")


def pulses_per_repetition_group(rotation_type: RotationType) -> int:
    if rotation_type == "PI":
        return 2
    if rotation_type == "PI_HALF":
        return 4
    raise ValueError(f"Unsupported rotation_type {rotation_type!r}.")


def _center_dense_amp_factors(
    minimum: float,
    maximum: float,
    uniform_count: int,
    step: float,
    density_power: float,
) -> np.ndarray:
    center = 1.0
    if center <= minimum or center >= maximum:
        amps = minimum + step * np.arange(uniform_count)
        if amps.size == 0 or not np.isclose(amps[-1], maximum):
            amps = np.append(amps, maximum)
        return amps.astype(float)

    total_count = uniform_count
    if not np.isclose(minimum + step * (uniform_count - 1), maximum):
        total_count += 1
    total_count = max(total_count, 3)

    left_span = center - minimum
    right_span = maximum - center
    left_count = max(2, int(round((total_count - 1) * left_span / (maximum - minimum))) + 1)
    right_count = max(2, total_count - left_count + 1)

    left_t = np.linspace(1, 0, left_count)
    right_t = np.linspace(0, 1, right_count)
    left = center - left_span * left_t**density_power
    right = center + right_span * right_t**density_power
    amps = np.concatenate([left, right[1:]])
    return np.unique(np.round(amps, decimals=12)).astype(float)
