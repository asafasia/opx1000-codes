from typing import Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


RotationType = Literal["PI", "PI_HALF"]


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
        amps = self.min_amp_factor + self.amp_factor_step * np.arange(count)
        if amps.size == 0 or not np.isclose(amps[-1], self.max_amp_factor):
            amps = np.append(amps, self.max_amp_factor)
        return amps.astype(float)

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
