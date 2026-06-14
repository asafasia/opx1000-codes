"""Build the QuAM wiring from a validated device profile."""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow direct execution and `python -m quam_config.wiring_lffem_mwfem`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize
from qualang_tools.wirer.connectivity.wiring_spec import (
    WiringFrequency,
    WiringIOType,
    WiringLineType,
)
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam

from profiles import ProfileError, load_profile
from quam_config import Quam


DEFAULT_PROFILE = "main"


def _controller_number(controller_name: str) -> int:
    """Convert profile controller names such as 'con1' to wirer controller IDs."""
    match = re.fullmatch(r"con(\d+)", controller_name)
    if match is None:
        raise ProfileError(
            f"Controller {controller_name!r} must use the form 'con<number>'"
        )
    return int(match.group(1))


def _qubit_reference(qubit_name: str) -> int:
    """Convert profile qubit names such as 'q9' to wirer qubit IDs."""
    match = re.fullmatch(r"q(\d+)", qubit_name)
    if match is None:
        raise ProfileError(f"Qubit {qubit_name!r} must use the form 'q<number>'")
    return int(match.group(1))


def _mw_fem_inventory(connectivity_profile: dict[str, Any]) -> dict[int, list[int]]:
    """Return MW-FEM slots grouped by controller."""
    inventory: dict[int, list[int]] = defaultdict(list)
    for controller_name, controller in connectivity_profile["controllers"].items():
        controller_number = _controller_number(controller_name)
        for fem_name, fem in controller["fems"].items():
            if fem["type"] != "mw_fem":
                raise ProfileError(
                    f"Unsupported FEM type {fem['type']!r} at "
                    f"{controller_name}/{fem_name}"
                )
            inventory[controller_number].append(int(fem_name))
    return dict(inventory)


def _resonator_lines(
    connections: dict[str, Any], active_qubits: list[str]
) -> dict[tuple[int, int, int, int], list[int]]:
    """Group qubits that share the same resonator input/output feedline."""
    lines: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for qubit_name in active_qubits:
        connection = connections[qubit_name]
        output = connection["resonator_output"]
        input_ = connection["resonator_input"]
        if output["controller"] != input_["controller"] or output["fem"] != input_["fem"]:
            raise ProfileError(
                f"{qubit_name!r} resonator input and output must use the same "
                "controller and MW FEM"
            )

        key = (
            _controller_number(output["controller"]),
            int(output["fem"]),
            int(input_["port"]),
            int(output["port"]),
        )
        lines[key].append(_qubit_reference(qubit_name))
    return dict(lines)


def _xy_lines(
    connections: dict[str, Any], active_qubits: list[str]
) -> dict[tuple[int, int, int], list[int]]:
    """Group qubits that share the same XY output line."""
    lines: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for qubit_name in active_qubits:
        output = connections[qubit_name]["xy_output"]
        key = (
            _controller_number(output["controller"]),
            int(output["fem"]),
            int(output["port"]),
        )
        lines[key].append(_qubit_reference(qubit_name))
    return dict(lines)


def create_profile_connectivity(
    profile: dict[str, Any],
) -> tuple[Connectivity, Instruments]:
    """Create allocated wirer connectivity from an already validated profile."""
    connectivity_profile = profile["connectivity"]
    qubit_names = list(profile["qubits"]["qubits"])
    connections = connectivity_profile["connections"]

    instruments = Instruments()
    for controller, slots in _mw_fem_inventory(connectivity_profile).items():
        instruments.add_mw_fem(controller=controller, slots=sorted(slots))

    connectivity = Connectivity()
    for (controller, fem, input_port, output_port), qubits in _resonator_lines(
        connections, qubit_names
    ).items():
        connectivity.add_resonator_line(
            qubits=qubits,
            constraints=mw_fem_spec(
                con=controller,
                slot=fem,
                in_port=input_port,
                out_port=output_port,
            ),
        )

    for (controller, fem, output_port), qubits in _xy_lines(
        connections, qubit_names
    ).items():
        constraints = mw_fem_spec(
            con=controller,
            slot=fem,
            out_port=output_port,
        )
        if len(qubits) == 1:
            connectivity.add_qubit_drive_lines(
                qubits=qubits,
                constraints=constraints,
            )
        else:
            connectivity.add_wiring_spec(
                frequency=WiringFrequency.RF,
                io_type=WiringIOType.OUTPUT,
                line_type=WiringLineType.DRIVE,
                triggered=False,
                constraints=constraints,
                elements=connectivity._make_qubit_elements(qubits),
                shared_line=True,
            )

    allocate_wiring(connectivity, instruments)
    return connectivity, instruments


def build_machine_from_profile(
    profile_name: str = DEFAULT_PROFILE, show_wiring: bool = True
) -> Quam:
    """Build and save wiring.json/state.json from a validated profile."""
    profile = load_profile(profile_name)
    network = profile["connectivity"]["network"]
    connectivity, instruments = create_profile_connectivity(profile)

    if show_wiring:
        visualize(
            connectivity.elements,
            available_channels=instruments.available_channels,
        )
        plt.show(block=False)

    machine = Quam()
    build_quam_wiring(
        connectivity=connectivity,
        host_ip=network["host"],
        cluster_name=network["cluster_name"],
        quam_instance=machine,
        port=network["port"],
    )
    build_quam(machine)
    return machine


def main() -> Quam:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile folder name under profiles/",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Build the QuAM without displaying the wiring diagram.",
    )
    args = parser.parse_args()
    return build_machine_from_profile(args.profile, show_wiring=not args.no_plot)


machine = None

if __name__ == "__main__":
    machine = main()
