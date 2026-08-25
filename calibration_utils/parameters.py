"""Repository-wide extensions to the shared qualibration parameters."""

from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters as _QubitsExperimentNodeParameters,
)


class QubitsExperimentNodeParameters(_QubitsExperimentNodeParameters):
    """Base parameters shared by qubit experiments in this repository."""

    use_readout_mitigation: float | bool = False
    """Readout-mitigation strength from 0 (off) to 1 (full matrix inversion)."""
