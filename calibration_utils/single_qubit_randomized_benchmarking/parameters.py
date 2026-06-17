from typing import Literal, Optional
import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


RBGateFamily = Literal["square", "drag", "cos", "cosine"]


class NodeSpecificParameters(RunnableParameters):
    use_state_discrimination: bool = True
    """Perform qubit state discrimination. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    gate_family: RBGateFamily = "square"
    """Pulse family used for the RB Clifford primitives: square, drag, cos, or cosine."""
    num_random_sequences: int = 30
    """Number of random RB sequences. Default is 300."""
    num_shots: int = 10
    """Number of averages. Default is 10."""
    max_circuit_depth: int = 204
    """Maximum circuit depth (number of Clifford gates). Default is 2048."""
    delta_clifford: int = 1
    """Delta clifford (number of Clifford gates between the RB sequences). Default is 20."""
    log_scale: bool = False
    """If True, use log scale depths: 1,2,4,8,16,32... up to max_circuit_depth. Default is True."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""
    fidelity_bootstrap_samples: int = 100
    """Bootstrap samples used to estimate the RB fidelity standard deviation. Set to 0 to disable."""
    fidelity_bootstrap_seed: Optional[int] = None
    """Seed for RB fidelity uncertainty bootstrap. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    def get_depths(self):
        """
        Generate an array of circuit depths based on the parameter configuration.

        This method produces a list of circuit depths depending on whether
        the `log_scale` flag is enabled:

        - If `log_scale` is True, depths follow a logarithmic progression:
          1, 2, 4, 8, 16, 32, ... up to `max_circuit_depth`.
        - If `log_scale` is False, depths are linearly spaced using
          `delta_clifford` until `max_circuit_depth`. The first value is
          always set to 1.

        Returns:
            numpy.ndarray: An array of circuit depths (integers).

        Examples:
            >>> params = Parameters(log_scale=True, max_circuit_depth=32)
            >>> params.get_depths()
            array([ 1,  2,  4,  8, 16, 32])

            >>> params = Parameters(log_scale=False,
            ...                     max_circuit_depth=10,
            ...                     delta_clifford=2)
            >>> params.get_depths()
            array([ 1,  2,  4,  6,  8, 10])
        """
        if self.max_circuit_depth < 1:
            raise ValueError("max_circuit_depth must be at least 1.")
        if self.delta_clifford < 1:
            raise ValueError("delta_clifford must be at least 1.")

        # Generate depth list based on log_scale parameter
        if self.log_scale:
            # Log scale: 1, 2, 4, 8, 16, 32, ... up to max_circuit_depth
            depths = [1]  # Start with depth 1
            current_depth = 2
            while current_depth <= self.max_circuit_depth:
                depths.append(current_depth)
                current_depth *= 2
            if depths[-1] != self.max_circuit_depth:
                depths.append(self.max_circuit_depth)
            depths = np.array(depths)
        else:
            # Linear scale using delta_clifford
            depths = np.arange(0, self.max_circuit_depth + 1, self.delta_clifford, dtype=int)
            depths[0] = 1  # Ensure we start with depth 1.
            depths = np.unique(depths)
            if depths[-1] != self.max_circuit_depth:
                depths = np.append(depths, self.max_circuit_depth)
        return depths
