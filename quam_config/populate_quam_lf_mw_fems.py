"""Apply a validated device profile to the generated QuAM state."""

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow direct execution and `python -m quam_config.populate_quam_lf_mw_fems`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quam.components.pulses import (
    DragCosinePulse,
    DragGaussianPulse,
    SquarePulse,
    SquareReadoutPulse,
)

from profiles import ProfileError, load_profile
from quam_config import Quam


DEFAULT_PROFILE = "main"


def _port_definition(
    connectivity: dict[str, Any], reference: dict[str, Any], direction: str
) -> dict[str, Any]:
    return connectivity["controllers"][reference["controller"]]["fems"][
        str(reference["fem"])
    ][direction][str(reference["port"])]


def _optional_assign(component: Any, attribute: str, value: Any) -> None:
    if value is not None:
        setattr(component, attribute, value)


def _create_pulse(
    pulse_name: str,
    pulse: dict[str, Any],
    qubit: Any,
    readout: dict[str, Any],
):
    common = {
        "length": pulse["length_ns"],
        "amplitude": pulse["amplitude"],
    }
    axis_angle = pulse.get("axis_angle_rad")

    if pulse["target"] == "resonator":
        return SquareReadoutPulse(
            **common,
            digital_marker=pulse.get("digital_marker", "ON"),
            axis_angle=axis_angle,
            threshold=readout["threshold"],
            integration_weights_angle=readout["integration_weights_angle_rad"],
        )

    if pulse["type"] in {"constant", "saturation"}:
        return SquarePulse(**common, axis_angle=axis_angle)

    if pulse["type"] == "drag":
        return DragGaussianPulse(
            **common,
            axis_angle=axis_angle or 0.0,
            sigma=pulse["sigma_ns"],
            alpha=pulse["beta"],
            anharmonicity=qubit.anharmonicity,
            detuning=pulse["detuning_hz"],
        )

    if pulse["type"] == "cosine":
        return DragCosinePulse(
            **common,
            axis_angle=axis_angle or 0.0,
            alpha=pulse.get("beta", 0.0),
            anharmonicity=qubit.anharmonicity,
            detuning=pulse.get("detuning_hz", 0.0),
        )

    raise ProfileError(f"Unsupported pulse type for {pulse_name!r}: {pulse['type']!r}")


def apply_profile(machine: Quam, profile: dict[str, Any]) -> Quam:
    """Apply frequencies, ports, readout settings, and pulses to a QuAM."""
    connectivity = profile["connectivity"]
    qubit_profiles = profile["qubits"]["qubits"]
    pulse_profiles = profile["pulses"]["pulses"]

    for qubit_name in profile["manifest"]["active_qubits"]:
        if qubit_name not in machine.qubits:
            raise ProfileError(
                f"Profile qubit {qubit_name!r} is missing from the generated QuAM. "
                "Run quam_config.wiring_lffem_mwfem first."
            )

        qubit = machine.qubits[qubit_name]
        settings = qubit_profiles[qubit_name]
        frequencies = settings["frequencies_hz"]
        transmon = settings["transmon"]
        readout = settings["readout"]
        connections = connectivity["connections"][qubit_name]

        xy_port = _port_definition(connectivity, connections["xy_output"], "outputs")
        rr_output = _port_definition(
            connectivity, connections["resonator_output"], "outputs"
        )
        rr_input = _port_definition(
            connectivity, connections["resonator_input"], "inputs"
        )

        qubit.f_01 = frequencies["qubit_f01"]
        qubit.f_12 = frequencies["qubit_f12"]
        qubit.anharmonicity = transmon["anharmonicity_hz"]
        qubit.grid_location = ",".join(str(value) for value in settings["grid_location"])
        _optional_assign(qubit, "T1", transmon["t1_ns"])
        _optional_assign(qubit, "T2ramsey", transmon["t2_ramsey_ns"])
        _optional_assign(qubit, "T2echo", transmon["t2_echo_ns"])

        qubit.xy.RF_frequency = frequencies["qubit_f01"]
        qubit.xy.opx_output.upconverter_frequency = frequencies["qubit_lo"]
        qubit.xy.opx_output.band = xy_port["band"]
        qubit.xy.opx_output.full_scale_power_dbm = xy_port["full_scale_power_dbm"]
        qubit.xy.opx_output.sampling_rate = xy_port["sampling_rate_hz"]

        qubit.resonator.f_01 = frequencies["resonator"]
        qubit.resonator.frequency_bare = frequencies["resonator_bare"]
        qubit.resonator.RF_frequency = frequencies["resonator"]
        qubit.resonator.time_of_flight = readout["time_of_flight_ns"]
        qubit.resonator.depletion_time = readout["depletion_time_ns"]
        qubit.resonator.opx_output.upconverter_frequency = frequencies["resonator_lo"]
        qubit.resonator.opx_output.band = rr_output["band"]
        qubit.resonator.opx_output.full_scale_power_dbm = rr_output[
            "full_scale_power_dbm"
        ]
        qubit.resonator.opx_output.sampling_rate = rr_output["sampling_rate_hz"]
        qubit.resonator.opx_input.band = rr_input["band"]
        qubit.resonator.opx_input.sampling_rate = rr_input["sampling_rate_hz"]

        qubit.xy.operations.clear()
        qubit.resonator.operations.clear()
        for operation_name, pulse_name in settings["operations"].items():
            pulse = _create_pulse(
                pulse_name,
                pulse_profiles[pulse_name],
                qubit,
                readout,
            )
            target_operations = (
                qubit.resonator.operations
                if pulse_profiles[pulse_name]["target"] == "resonator"
                else qubit.xy.operations
            )
            target_operations[operation_name] = pulse

    machine.active_qubit_names = profile["manifest"]["active_qubits"]
    return machine


def populate_quam(profile_name: str = DEFAULT_PROFILE, save: bool = True) -> Quam:
    """Load the generated QuAM, apply a profile, and optionally save it."""
    profile = load_profile(profile_name)
    machine = apply_profile(Quam.load(), profile)
    if save:
        machine.save()
    return machine


def main() -> Quam:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile folder name under profiles/",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Apply and validate the profile without writing state.json.",
    )
    args = parser.parse_args()
    machine = populate_quam(args.profile, save=not args.no_save)
    print(
        f"Applied profile {args.profile!r} to "
        f"{len(machine.active_qubit_names)} active qubit(s)."
    )
    return machine


if __name__ == "__main__":
    main()
