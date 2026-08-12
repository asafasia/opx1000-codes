"""QUA macros for threshold and nearest-center state discrimination."""

import math
from collections.abc import Sequence
from typing import Literal

from qm.qua import Math, Cast, assign, declare, elif_, fixed, if_, wait, while_


StateOneWhen = Literal["above_threshold", "below_threshold"]
Discriminator = Literal["quam", "nearest_center"]
DEFAULT_DISTANCE_SCALE = 128.0


def _validated_centers(
    centers: Sequence[Sequence[float]],
    num_states: int,
) -> tuple[tuple[float, float], ...]:
    if num_states not in {2, 3}:
        raise ValueError(f"num_states must be 2 or 3, got {num_states!r}")
    if not isinstance(centers, Sequence) or len(centers) < num_states:
        raise ValueError(
            f"Nearest-center discrimination needs at least {num_states} IQ centers"
        )

    selected = []
    # Index instead of slicing: QuAM's list wrapper supports indexing but
    # intentionally prevents constructing a sliced child with a new parent.
    for state_index in range(num_states):
        center = centers[state_index]
        if not isinstance(center, Sequence) or len(center) != 2:
            raise ValueError(
                f"IQ center {state_index} must contain exactly [I, Q]"
            )
        i_center, q_center = center
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (i_center, q_center)
        ):
            raise ValueError(f"IQ center {state_index} must contain finite numbers")
        selected.append((float(i_center), float(q_center)))
    return tuple(selected)


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


def discriminate_nearest_center(
    i_quadrature,
    q_quadrature,
    centers: Sequence[Sequence[float]],
    *,
    num_states: int | None = None,
    distance_scale: float = DEFAULT_DISTANCE_SCALE,
):
    """Return the QUA state index of the closest calibrated IQ center.

    Centers are ordered as G/E for two-state readout and G/E/F for three-state
    readout. Squared Euclidean distances are compared because taking a square
    root would not change the closest-center result. The deltas are multiplied
    by a common scale before squaring so nearby centers remain distinguishable
    at QUA fixed-point precision; the scale does not change which center is
    closest.
    """
    if num_states is None:
        num_states = len(centers)
    if not isinstance(distance_scale, (int, float)) or not math.isfinite(
        distance_scale
    ) or distance_scale <= 0:
        raise ValueError("distance_scale must be a positive finite number")
    selected_centers = _validated_centers(centers, num_states)
    squared_distances = declare(fixed, size=num_states)

    for state_index, (i_center, q_center) in enumerate(selected_centers):
        i_delta = (i_quadrature - i_center) * float(distance_scale)
        q_delta = (q_quadrature - q_center) * float(distance_scale)
        assign(
            squared_distances[state_index],
            i_delta * i_delta + q_delta * q_delta,
        )

    return Math.argmin(squared_distances)


def readout_state_nearest_center(
    qubit,
    state,
    *,
    num_states: int = 2,
    centers: Sequence[Sequence[float]] | None = None,
    pulse_name: str | None = None,
) -> None:
    """Measure and assign ``state`` using the closest IQ blob center."""
    if centers is None:
        centers = getattr(qubit.resonator, "gef_centers", None)
    if centers is None:
        raise ValueError(
            f"{qubit.name} has no calibrated IQ centers; run IQ blobs and apply "
            "its profile update first"
        )
    selected_centers = _validated_centers(centers, num_states)
    if pulse_name is None:
        pulse_name = "readout_GEF" if num_states == 3 else "readout"

    i_quadrature = declare(fixed)
    q_quadrature = declare(fixed)
    uses_gef_frequency = num_states == 3
    if uses_gef_frequency:
        qubit.resonator.update_frequency(
            int(
                qubit.resonator.intermediate_frequency
                + qubit.resonator.GEF_frequency_shift
            )
        )
    qubit.resonator.measure(
        pulse_name,
        qua_vars=(i_quadrature, q_quadrature),
    )
    if uses_gef_frequency:
        qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)
    assign(
        state,
        discriminate_nearest_center(
            i_quadrature,
            q_quadrature,
            selected_centers,
            num_states=num_states,
        ),
    )
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)


def readout_state_configured(
    qubit,
    state,
    *,
    num_states: int = 2,
    pulse_name: str | None = None,
    discriminator: Discriminator | None = None,
) -> None:
    """Dispatch readout through the profile-selected state discriminator.

    ``discriminator`` is mainly useful for tests and one-off experiments. When
    omitted, the global profile choice copied onto the resonator is used.
    """
    if num_states not in {2, 3}:
        raise ValueError(f"num_states must be 2 or 3, got {num_states!r}")
    if discriminator is None:
        discriminator = getattr(qubit.resonator, "readout_discriminator", "quam")

    if discriminator == "nearest_center":
        readout_state_nearest_center(
            qubit,
            state,
            num_states=num_states,
            pulse_name=pulse_name,
        )
        return
    if discriminator == "quam":
        if num_states == 2:
            qubit.readout_state(state, pulse_name=pulse_name or "readout")
        else:
            qubit.readout_state_gef(
                state,
                pulse_name=pulse_name or "readout_GEF",
            )
        return
    raise ValueError(
        "discriminator must be 'quam' or 'nearest_center', "
        f"got {discriminator!r}"
    )


def active_reset_configured(
    qubit,
    *,
    num_states: int = 2,
    max_attempts: int = 15,
    pulse_name: str | None = None,
    discriminator: Discriminator | None = None,
    ge_operation: str = "x180",
    ef_operation: str = "EF_x180",
):
    """Actively reset G/E or G/E/F using the configured discriminator.

    Each attempt measures the state, then applies E->G for state 1 or
    F->E->G for state 2. The loop is bounded even if readout repeatedly reports
    an excited state.

    Returns:
        ``(state, attempts)`` QUA variables containing the last discriminated
        state and the number of measurements performed.
    """
    if num_states not in {2, 3}:
        raise ValueError(f"num_states must be 2 or 3, got {num_states!r}")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if ge_operation not in qubit.xy.operations:
        raise ValueError(
            f"{qubit.name} does not define GE reset operation {ge_operation!r}"
        )
    if num_states == 3 and ef_operation not in qubit.xy.operations:
        raise ValueError(
            f"{qubit.name} does not define EF reset operation {ef_operation!r}"
        )

    state = declare(int)
    attempts = declare(int)
    # The macro can be called inside a QUA loop. A declaration initializer runs
    # only once when the program starts, while this assignment resets the
    # counter on every macro invocation/shot.
    assign(attempts, 1)
    ge_frequency = int(qubit.xy.intermediate_frequency)

    def apply_correction() -> None:
        with if_(state == 1):
            qubit.xy.update_frequency(ge_frequency, keep_phase=True)
            qubit.xy.play(ge_operation)
        if num_states == 3:
            with elif_(state == 2):
                qubit.xy.update_frequency(
                    int(qubit.xy.intermediate_frequency - qubit.anharmonicity),
                    keep_phase=True,
                )
                qubit.xy.play(ef_operation)
                qubit.xy.update_frequency(ge_frequency, keep_phase=True)
                qubit.xy.play(ge_operation)

    qubit.align()
    readout_state_configured(
        qubit,
        state,
        num_states=num_states,
        pulse_name=pulse_name,
        discriminator=discriminator,
    )
    qubit.align()
    apply_correction()

    with while_((state > 0) & (attempts < max_attempts)):
        qubit.align()
        readout_state_configured(
            qubit,
            state,
            num_states=num_states,
            pulse_name=pulse_name,
            discriminator=discriminator,
        )
        qubit.align()
        apply_correction()
        assign(attempts, attempts + 1)

    qubit.align()
    return state, attempts


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
