from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50000
    """Number of shots to perform. Default is 50000."""
    start_amp: float = 0.5
    """Start amplitude. Default is 0.5."""
    end_amp: float = 2.5
    """End amplitude. Default is 2.59."""
    num_amps: int = 10
    """Number of amplitudes to sweep. Default is 10."""
    outliers_threshold: float = 0.98
    """Outliers threshold. Default is 0.98."""
    plot_raw: bool = False
    """Plot raw data. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
