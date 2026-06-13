"""QUA macros for threshold-based qubit-state discrimination."""

from typing import Literal

from qm.qua import Cast, assign, declare, fixed


StateOneWhen = Literal["above_threshold", "below_threshold"]


def discriminate_i(i_quadrature, threshold, state_1_when: StateOneWhen = "above_threshold"):
    """Return a QUA integer expression that is 0 or 1 based on an I threshold.

    Args:
        i_quadrature: Measured QUA fixed variable containing the rotated I value.
        threshold: Python number or QUA fixed threshold in demodulation units.
        state_1_when: Select whether state 1 lies above or below the threshold.

    Returns:
        A QUA integer expression equal to 0 or 1.
    """
    if state_1_when == "above_threshold":
        return Cast.to_int(i_quadrature > threshold)
    if state_1_when == "below_threshold":
        return Cast.to_int(i_quadrature < threshold)
    raise ValueError(
        "state_1_when must be 'above_threshold' or 'below_threshold', "
        f"got {state_1_when!r}"
    )


def readout_state(
    qubit,
    threshold=None,
    pulse_name: str = "readout",
    state_1_when: StateOneWhen = "above_threshold",
):
    """Measure a qubit and return its discriminated state together with I and Q.

    If no threshold is provided, the threshold stored on the selected readout
    pulse is used.
    """
    i_quadrature = declare(fixed)
    q_quadrature = declare(fixed)
    state = declare(int)

    if threshold is None:
        threshold = qubit.resonator.operations[pulse_name].threshold

    qubit.resonator.measure(
        pulse_name,
        qua_vars=(i_quadrature, q_quadrature),
    )
    assign(state, discriminate_i(i_quadrature, threshold, state_1_when))
    return state, i_quadrature, q_quadrature
