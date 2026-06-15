"""Reusable QUA macros for a classical analog Ising machine."""

from qm.qua import Cast, assign, else_, if_


def assign_state_from_signal(signal, threshold, state):
    """Classify an analog-like QUA value as binary state 0 or 1.

    Example mapping:
        signal = 1.0 -> state 1
        signal = 0.5 -> state 0
        threshold = 0.75

    Args:
        signal: QUA fixed variable containing an assigned or measured value.
        threshold: Python number or QUA fixed threshold.
        state: QUA int variable that receives 0 or 1.
    """
    with if_(signal > threshold):
        assign(state, 1)
    with else_():
        assign(state, 0)


def state_to_spin(state, spin):
    """Convert binary state 0/1 to Ising spin +1/-1."""
    with if_(state == 0):
        assign(spin, 1)
    with else_():
        assign(spin, -1)


def assign_spin_from_signal(signal, threshold, state, spin):
    """Classify a signal and assign both its binary state and Ising spin."""
    assign_state_from_signal(signal, threshold, state)
    state_to_spin(state, spin)


def calculate_single_spin_energy(spin, field, energy):
    """Calculate E = -h*s for one classical spin.

    Args:
        spin: QUA int spin variable containing -1 or +1.
        field: Python number or QUA fixed longitudinal field h.
        energy: QUA fixed variable that receives the energy.
    """
    assign(energy, Cast.mul_fixed_by_int(-field, spin))


def calculate_flip_energy(spin, field, delta_energy):
    """Calculate the energy change dE = 2*h*s caused by flipping one spin."""
    assign(delta_energy, 2.0 * Cast.mul_fixed_by_int(field, spin))


def zero_temperature_update(state, spin, delta_energy, flipped):
    """Flip the spin only when doing so lowers its energy.

    Args:
        state: QUA int binary state variable containing 0 or 1.
        spin: QUA int Ising spin variable containing -1 or +1.
        delta_energy: QUA fixed variable containing the flip energy change.
        flipped: QUA int variable that receives 1 when a flip occurred.
    """
    with if_(delta_energy < 0):
        assign(state, 1 - state)
        assign(spin, -spin)
        assign(flipped, 1)
    with else_():
        assign(flipped, 0)
