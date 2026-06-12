"""Load and validate versioned experiment profiles."""

import json
from pathlib import Path
from typing import Any


PROFILES_ROOT = Path(__file__).resolve().parent
SUPPORTED_SCHEMA_VERSION = 1
PULSE_TYPES = {"constant", "drag", "cosine", "saturation"}
STATE_1_RULES = {"above_threshold", "below_threshold"}
MW_FEM_BAND_RANGES_HZ = {
    1: (50e6, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}


class ProfileError(ValueError):
    """Raised when a profile is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ProfileError(f"Profile file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ProfileError(f"Profile file must contain a JSON object: {path}")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def _validate_version(document: dict[str, Any], document_name: str) -> None:
    _require(
        document.get("schema_version") == SUPPORTED_SCHEMA_VERSION,
        f"{document_name}.schema_version must be {SUPPORTED_SCHEMA_VERSION}",
    )


def _validate_pulses(pulses_document: dict[str, Any]) -> None:
    pulses = pulses_document.get("pulses")
    _require(isinstance(pulses, dict) and pulses, "pulses.json must define pulses")

    for name, pulse in pulses.items():
        _require(isinstance(pulse, dict), f"Pulse {name!r} must be an object")
        pulse_type = pulse.get("type")
        target = pulse.get("target")
        _require(pulse_type in PULSE_TYPES, f"Pulse {name!r} has unsupported type {pulse_type!r}")
        _require(target in {"qubit", "resonator"}, f"Pulse {name!r} has invalid target {target!r}")
        _require(isinstance(pulse.get("amplitude"), (int, float)), f"Pulse {name!r} needs amplitude")
        _require(pulse["amplitude"] != 0, f"Pulse {name!r} amplitude cannot be zero")
        _require(isinstance(pulse.get("length_ns"), int) and pulse["length_ns"] > 0, f"Pulse {name!r} needs positive integer length_ns")
        _require(target != "resonator" or pulse_type == "constant", f"Readout pulse {name!r} must use type 'constant'")

        if pulse_type == "drag":
            _require(isinstance(pulse.get("sigma_ns"), (int, float)) and pulse["sigma_ns"] > 0, f"DRAG pulse {name!r} needs positive sigma_ns")
            _require(isinstance(pulse.get("beta"), (int, float)), f"DRAG pulse {name!r} needs beta")
            _require(isinstance(pulse.get("detuning_hz"), (int, float)), f"DRAG pulse {name!r} needs detuning_hz")


def _get_port(connectivity: dict[str, Any], reference: dict[str, Any], direction: str, label: str) -> dict[str, Any]:
    try:
        controller = connectivity["controllers"][reference["controller"]]
        fem = controller["fems"][str(reference["fem"])]
        return fem[direction][str(reference["port"])]
    except (KeyError, TypeError) as exc:
        raise ProfileError(f"{label} references an undefined {direction[:-1]}") from exc


def _validate_mw_port(port: dict[str, Any], label: str) -> None:
    band = port.get("band")
    lo_frequency = port.get("lo_frequency_hz")
    _require(band in MW_FEM_BAND_RANGES_HZ, f"{label} has unsupported MW-FEM band {band!r}")
    _require(
        isinstance(lo_frequency, (int, float)) and lo_frequency > 0,
        f"{label} needs positive lo_frequency_hz",
    )
    minimum, maximum = MW_FEM_BAND_RANGES_HZ[band]
    _require(
        minimum <= lo_frequency <= maximum,
        f"{label} LO {lo_frequency} Hz is outside band {band} range "
        f"[{minimum:g}, {maximum:g}] Hz",
    )


def _validate_connectivity(connectivity: dict[str, Any], qubits_document: dict[str, Any]) -> None:
    connections = connectivity.get("connections")
    qubits = qubits_document.get("qubits")
    _require(isinstance(connections, dict), "connectivity.json must define connections")

    for qubit_name in qubits:
        _require(qubit_name in connections, f"Qubit {qubit_name!r} has no connectivity entry")
        connection = connections[qubit_name]
        xy_output = _get_port(connectivity, connection["xy_output"], "outputs", f"{qubit_name}.xy_output")
        resonator_output = _get_port(
            connectivity,
            connection["resonator_output"],
            "outputs",
            f"{qubit_name}.resonator_output",
        )
        resonator_input = _get_port(
            connectivity,
            connection["resonator_input"],
            "inputs",
            f"{qubit_name}.resonator_input",
        )
        frequencies = qubits[qubit_name]["frequencies_hz"]
        _validate_mw_port(xy_output, f"Qubit {qubit_name!r} XY output")
        _validate_mw_port(resonator_output, f"Qubit {qubit_name!r} resonator output")
        _validate_mw_port(resonator_input, f"Qubit {qubit_name!r} resonator input")
        _require(
            resonator_input.get("lo_frequency_hz") == resonator_output["lo_frequency_hz"],
            f"Qubit {qubit_name!r} resonator input and output LOs do not match",
        )
        _require(
            abs(frequencies["qubit_f01"] - xy_output["lo_frequency_hz"]) <= 400e6,
            f"Qubit {qubit_name!r} qubit IF exceeds 400 MHz: "
            f"RF={frequencies['qubit_f01']} Hz, connectivity XY LO={xy_output['lo_frequency_hz']} Hz",
        )
        _require(
            abs(frequencies["resonator"] - resonator_output["lo_frequency_hz"]) <= 400e6,
            f"Qubit {qubit_name!r} resonator IF exceeds 400 MHz: "
            f"RF={frequencies['resonator']} Hz, connectivity output LO={resonator_output['lo_frequency_hz']} Hz",
        )


def _validate_qubits(qubits_document: dict[str, Any], pulses_document: dict[str, Any]) -> None:
    qubits = qubits_document.get("qubits")
    pulses = pulses_document["pulses"]
    _require(isinstance(qubits, dict) and qubits, "qubits.json must define qubits")

    for name, qubit in qubits.items():
        frequencies = qubit.get("frequencies_hz", {})
        operations = qubit.get("operations", {})
        readout = qubit.get("readout", {})
        transmon = qubit.get("transmon", {})

        for frequency_name in ("qubit_f01", "resonator"):
            _require(
                isinstance(frequencies.get(frequency_name), (int, float)) and frequencies[frequency_name] > 0,
                f"Qubit {name!r} needs positive frequencies_hz.{frequency_name}",
            )

        _require(readout.get("state_1_when") in STATE_1_RULES, f"Qubit {name!r} has invalid readout.state_1_when")
        _require(
            isinstance(transmon.get("thermalization_time_ns"), int)
            and transmon["thermalization_time_ns"] > 0,
            f"Qubit {name!r} needs positive integer transmon.thermalization_time_ns",
        )
        _require(isinstance(operations, dict) and operations, f"Qubit {name!r} must define operations")

        for operation_name, pulse_name in operations.items():
            _require(pulse_name in pulses, f"Qubit {name!r} operation {operation_name!r} references unknown pulse {pulse_name!r}")
            target = pulses[pulse_name]["target"]
            expected_target = "resonator" if operation_name.startswith("readout") else "qubit"
            _require(target == expected_target, f"Qubit {name!r} operation {operation_name!r} must target {expected_target}")


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate a loaded profile, raising ProfileError on the first issue."""
    manifest = profile["manifest"]
    connectivity = profile["connectivity"]
    qubits = profile["qubits"]
    pulses = profile["pulses"]

    for name, document in profile.items():
        _validate_version(document, name)

    _validate_pulses(pulses)
    _validate_qubits(qubits, pulses)
    _validate_connectivity(connectivity, qubits)

    active_qubits = manifest.get("active_qubits")
    _require(isinstance(active_qubits, list) and active_qubits, "profile.json must define active_qubits")
    for qubit_name in active_qubits:
        _require(qubit_name in qubits["qubits"], f"Active qubit {qubit_name!r} is undefined")


def load_profile(name: str = "main") -> dict[str, Any]:
    """Load and validate a named profile from the profiles directory."""
    profile_directory = PROFILES_ROOT / name
    manifest = _read_json(profile_directory / "profile.json")
    _validate_version(manifest, "manifest")
    files = manifest.get("files", {})

    profile = {
        "manifest": manifest,
        "connectivity": _read_json(profile_directory / files.get("connectivity", "connectivity.json")),
        "qubits": _read_json(profile_directory / files.get("qubits", "qubits.json")),
        "pulses": _read_json(profile_directory / files.get("pulses", "pulses.json")),
    }
    validate_profile(profile)
    return profile
