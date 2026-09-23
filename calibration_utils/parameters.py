"""Repository-wide extensions to the shared qualibration parameters."""

from typing import Literal

from pydantic import Field, field_validator
from calibration_utils.state_acquisition import StateAcquisitionParameters

from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters as _QubitsExperimentNodeParameters,
)


class QubitsExperimentNodeParameters(_QubitsExperimentNodeParameters, StateAcquisitionParameters):
    """Base parameters shared by qubit experiments in this repository."""

    use_readout_mitigation: float | bool = False
    """Readout-mitigation strength from 0 (off) to 1 (full matrix inversion)."""

    readout_states: list[Literal["g", "e", "f"]] = Field(default_factory=lambda: ["g", "e"])
    """Readout basis: ['g', 'e'] uses readout; ['g', 'e', 'f'] uses readout_GEF."""
    active_reset_max_attempts: int = Field(default=15, ge=1)
    """Maximum measurements per active reset; its basis follows readout_states."""

    @field_validator("readout_states")
    @classmethod
    def validate_readout_states(cls, value):
        if value not in (["g", "e"], ["g", "e", "f"]):
            raise ValueError("readout_states must be ['g', 'e'] or ['g', 'e', 'f'] in that order")
        return value

    @property
    def readout_operation(self) -> str:
        return "readout_GEF" if len(self.readout_states) == 3 else "readout"
