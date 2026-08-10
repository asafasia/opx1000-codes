"""Repository-wide extensions to the shared qualibration parameters."""

from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters as _QubitsExperimentNodeParameters,
)


class QubitsExperimentNodeParameters(_QubitsExperimentNodeParameters):
    """Base parameters shared by qubit experiments in this repository."""

    use_readout_mitigation: bool = False
    """Correct discriminated state populations using the IQ-blobs assignment matrix."""
