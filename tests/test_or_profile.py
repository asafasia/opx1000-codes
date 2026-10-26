"""Exact chip-label selection and QuAM wiring, without hardware access."""

from copy import deepcopy

import pytest

from profiles import load_profile
from quam_config.create_machine_from_profile import create_machine_from_profile


NAMES = [f"q{i}{suffix}" for suffix in ("b", "a") for i in range(1, 4)]


def test_or_profile_contains_six_independent_entries():
    profile = load_profile("Or_profile")
    assert profile["manifest"]["active_qubits"] == ["q3b"]
    for document, section in (("qubits", "qubits"), ("pulses", "pulses"),
                              ("metrics", "qubits"), ("connectivity", "connections")):
        entries = profile[document][section]
        assert set(entries) == set(NAMES)
        before = deepcopy(entries["q2b"])
        entries["q2a"].clear()
        assert entries["q2b"] == before


@pytest.mark.parametrize("name", NAMES)
def test_chip_label_builds_selected_machine_and_config(name):
    profile = load_profile("Or_profile", qubit=name)
    machine = create_machine_from_profile("Or_profile", save=False, qubit=name)
    assert list(machine.qubits) == [name]
    assert machine.active_qubit_names == [name]
    qubit = machine.qubits[name]
    assert qubit.name == name
    assert qubit.f_01 == profile["qubits"]["qubits"][name]["frequencies_hz"]["qubit_f01"]
    connection = profile["connectivity"]["connections"][name]
    assert qubit.xy.opx_output.port_id == connection["xy_output"]["port"]
    assert qubit.resonator.opx_output.port_id == connection["resonator_output"]["port"]
    assert qubit.resonator.opx_input.port_id == connection["resonator_input"]["port"]
    config = machine.generate_config()
    assert f"{name}.xy" in config["elements"]
    assert f"{name}.resonator" in config["elements"]


def test_all_six_entries_build_without_name_collisions():
    machine = create_machine_from_profile("Or_profile", save=False)
    assert set(machine.qubits) == set(NAMES)
    assert set(machine.wiring["qubits"]) == set(NAMES)
    assert machine.active_qubit_names == ["q3b"]
    config = machine.generate_config()
    assert all(f"{name}.xy" in config["elements"] for name in NAMES)
