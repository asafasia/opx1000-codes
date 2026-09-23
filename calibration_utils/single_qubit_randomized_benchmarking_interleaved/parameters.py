from typing import Literal
from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters as RBParameters


class Parameters(RBParameters):
    mode: Literal["interleaved"] = "interleaved"
    num_random_sequences: int = 100
    num_shots: int = 20
    max_circuit_depth: int = 1000
    delta_clifford: int = 20
    use_state_discrimination: bool = False


def get_interleaved_gate_name(gate_index: int) -> str:
    """Return the name of the gate based on its Clifford index."""
    if gate_index == 0:
        return "I"
    elif gate_index == 1:
        return "x180"
    elif gate_index == 2:
        return "y180"
    elif gate_index == 12:
        return "x90"
    elif gate_index == 13:
        return "-x90"
    elif gate_index == 14:
        return "y90"
    elif gate_index == 15:
        return "-y90"
    else:
        raise ValueError(f"Interleaved gate index {gate_index} doesn't correspond to a single operation")


def get_interleaved_gate_index(gate_operation) -> int:
    """Return the Clifford gate index corresponding to the specified gate name."""
    if gate_operation == "I":
        return 0
    elif gate_operation == "x180":
        return 1
    elif gate_operation == "y180":
        return 2
    elif gate_operation == "x90":
        return 12
    elif gate_operation == "-x90":
        return 13
    elif gate_operation == "y90":
        return 14
    elif gate_operation == "-y90":
        return 15
    else:
        raise ValueError(f"Gate operation {gate_operation} not recognized")
