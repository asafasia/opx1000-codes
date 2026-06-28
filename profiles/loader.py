"""Load and validate versioned experiment profiles."""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PROFILES_ROOT = Path(__file__).resolve().parent
SUPPORTED_SCHEMA_VERSION = 1
PULSE_TYPES = {"constant", "drag", "cosine", "saturation"}
STATE_1_RULES = {"above_threshold", "below_threshold"}
FLUX_POINTS = {"joint", "independent", "min", "arbitrary", "zero"}
MAX_PROFILE_PULSE_AMPLITUDE = 0.7
MW_FEM_BAND_RANGES_HZ = {
    1: (50e6, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}
MW_FEM_SHARED_LO_OUTPUT_PAIRS = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
MW_FEM_MAX_IF_HZ = 500e6
_active_profile: "Profile | None" = None


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


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    json.loads(rendered)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(rendered)
    temporary_path.replace(path)


def _profile_directory(root: Path, name: str) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ProfileError(f"Invalid profile name: {name!r}")
    directory = (root / name).resolve()
    try:
        directory.relative_to(root.resolve())
    except ValueError as exc:
        raise ProfileError(f"Profile {name!r} is outside profiles/.") from exc
    if not directory.is_dir():
        raise ProfileError(f"Profile directory does not exist: {directory}")
    return directory


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def _validate_version(document: dict[str, Any], document_name: str) -> None:
    _require(
        document.get("schema_version") == SUPPORTED_SCHEMA_VERSION,
        f"{document_name}.schema_version must be {SUPPORTED_SCHEMA_VERSION}",
    )


def _validate_pulses(pulses_document: dict[str, Any]) -> None:
    pulses_by_qubit = pulses_document.get("pulses")
    _require(isinstance(pulses_by_qubit, dict) and pulses_by_qubit, "pulses.json must define pulses")

    for qubit_name, pulses in pulses_by_qubit.items():
        _require(isinstance(pulses, dict) and pulses, f"Qubit {qubit_name!r} must define pulses")
        for name, pulse in pulses.items():
            _validate_pulse(f"{qubit_name}.{name}", pulse)


def _validate_pulse(name: str, pulse: Any) -> None:
    _require(isinstance(pulse, dict), f"Pulse {name!r} must be an object")
    pulse_type = pulse.get("type")
    target = pulse.get("target")
    _require(pulse_type in PULSE_TYPES, f"Pulse {name!r} has unsupported type {pulse_type!r}")
    _require(target in {"qubit", "resonator"}, f"Pulse {name!r} has invalid target {target!r}")
    _require(isinstance(pulse.get("amplitude"), (int, float)), f"Pulse {name!r} needs amplitude")
    _require(pulse["amplitude"] != 0, f"Pulse {name!r} amplitude cannot be zero")
    _require(
        abs(pulse["amplitude"]) <= MAX_PROFILE_PULSE_AMPLITUDE,
        f"Pulse {name!r} amplitude is too high: {pulse['amplitude']!r}. "
        f"Maximum allowed absolute amplitude is {MAX_PROFILE_PULSE_AMPLITUDE}.",
    )
    _require(isinstance(pulse.get("length_ns"), int) and pulse["length_ns"] > 0, f"Pulse {name!r} needs positive integer length_ns")
    _require(target != "resonator" or pulse_type == "constant", f"Readout pulse {name!r} must use type 'constant'")

    if target == "resonator":
        integration_weights = pulse.get("integration_weights")
        _require(
            isinstance(integration_weights, list) and integration_weights,
            f"Readout pulse {name!r} needs integration_weights",
        )
        _require(
            all(
                isinstance(segment, list)
                and len(segment) == 2
                and isinstance(segment[0], (int, float))
                and isinstance(segment[1], int)
                and segment[1] > 0
                for segment in integration_weights
            ),
            f"Readout pulse {name!r} integration_weights must contain [weight, length_ns] segments",
        )
        _require(
            sum(segment[1] for segment in integration_weights) == pulse["length_ns"],
            f"Readout pulse {name!r} integration_weights must span length_ns",
        )

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


def _validate_mw_input_port(port: dict[str, Any], label: str) -> None:
    band = port.get("band")
    _require(band in MW_FEM_BAND_RANGES_HZ, f"{label} has unsupported MW-FEM band {band!r}")
    _require(
        "lo_frequency_hz" not in port,
        f"{label} must not define lo_frequency_hz; it is derived from the resonator output",
    )


def _validate_lf_output_port(port: dict[str, Any], label: str) -> None:
    sampling_rate = port.get("sampling_rate_hz")
    _require(
        isinstance(sampling_rate, (int, float)) and sampling_rate > 0,
        f"{label} needs positive sampling_rate_hz",
    )
    offset = port.get("offset_v")
    _require(
        offset is None or isinstance(offset, (int, float)),
        f"{label} offset_v must be numeric or null",
    )


def _validate_shared_lo_output_pairs(connectivity: dict[str, Any]) -> None:
    """Require configured MW-FEM output pairs to use their shared physical LO."""
    for controller_name, controller in connectivity["controllers"].items():
        for fem_name, fem in controller["fems"].items():
            if fem["type"] != "mw_fem":
                continue
            outputs = fem.get("outputs", {})
            for first, second in MW_FEM_SHARED_LO_OUTPUT_PAIRS:
                first_port = outputs.get(str(first))
                second_port = outputs.get(str(second))
                if first_port is None or second_port is None:
                    continue
                _require(
                    first_port.get("lo_frequency_hz") == second_port.get("lo_frequency_hz"),
                    f"{controller_name} MW-FEM {fem_name} outputs {first} and {second} "
                    "share an LO and must use the same lo_frequency_hz",
                )


def _global_flux_output_reference(
    connectivity: dict[str, Any], connection: dict[str, Any]
) -> dict[str, Any] | None:
    z_output = connection.get("z_output")
    if z_output is None:
        return None
    if "global_line" not in z_output:
        return z_output

    lines = connectivity.get("global_flux_lines", {})
    try:
        return lines[z_output["global_line"]].get("output")
    except KeyError as exc:
        raise ProfileError(
            f"Global flux line {z_output['global_line']!r} is undefined"
        ) from exc


def _validate_connectivity(connectivity: dict[str, Any], qubits_document: dict[str, Any]) -> None:
    connections = connectivity.get("connections")
    qubits = qubits_document.get("qubits")
    _require(isinstance(connections, dict), "connectivity.json must define connections")
    _validate_shared_lo_output_pairs(connectivity)

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
        _validate_mw_input_port(resonator_input, f"Qubit {qubit_name!r} resonator input")
        _require(
            resonator_input["band"] == resonator_output["band"],
            f"Qubit {qubit_name!r} resonator input and output bands do not match",
        )
        _require(
            abs(frequencies["qubit_f01"] - xy_output["lo_frequency_hz"]) <= MW_FEM_MAX_IF_HZ,
            f"Qubit {qubit_name!r} qubit IF exceeds {MW_FEM_MAX_IF_HZ / 1e6:g} MHz: "
            f"RF={frequencies['qubit_f01']} Hz, connectivity XY LO={xy_output['lo_frequency_hz']} Hz",
        )
        _require(
            abs(frequencies["resonator"] - resonator_output["lo_frequency_hz"]) <= MW_FEM_MAX_IF_HZ,
            f"Qubit {qubit_name!r} resonator IF exceeds {MW_FEM_MAX_IF_HZ / 1e6:g} MHz: "
            f"RF={frequencies['resonator']} Hz, connectivity output LO={resonator_output['lo_frequency_hz']} Hz",
        )
        z_output_reference = _global_flux_output_reference(connectivity, connection)
        if z_output_reference is not None:
            z_output = _get_port(
                connectivity,
                z_output_reference,
                "outputs",
                f"{qubit_name}.z_output",
            )
            _validate_lf_output_port(z_output, f"Qubit {qubit_name!r} Z output")


def _validate_qubits(qubits_document: dict[str, Any], pulses_document: dict[str, Any]) -> None:
    qubits = qubits_document.get("qubits")
    pulses = pulses_document["pulses"]
    _require(isinstance(qubits, dict) and qubits, "qubits.json must define qubits")

    for name, qubit in qubits.items():
        _require(
            "enabled" not in qubit,
            f"Qubit {name!r} must not define enabled; use profile.json active_qubits",
        )
        _require(name in pulses, f"Qubit {name!r} has no pulse definitions")
        qubit_pulses = pulses[name]
        frequencies = qubit.get("frequencies_hz", {})
        operations = qubit.get("operations", {})
        readout = qubit.get("readout", {})
        transmon = qubit.get("transmon", {})
        flux = qubit.get("flux")

        for frequency_name in ("qubit_f01", "resonator"):
            _require(
                isinstance(frequencies.get(frequency_name), (int, float)) and frequencies[frequency_name] > 0,
                f"Qubit {name!r} needs positive frequencies_hz.{frequency_name}",
            )

        _require(readout.get("state_1_when") in STATE_1_RULES, f"Qubit {name!r} has invalid readout.state_1_when")
        for threshold_name in ("threshold", "rus_exit_threshold"):
            _require(
                isinstance(readout.get(threshold_name), (int, float)),
                f"Qubit {name!r} needs numeric readout.{threshold_name}",
            )
        for timing_name in ("time_of_flight_ns", "depletion_time_ns"):
            _require(
                isinstance(readout.get(timing_name), int) and readout[timing_name] > 0,
                f"Qubit {name!r} needs positive integer readout.{timing_name}",
            )
        _require(
            isinstance(readout.get("smearing_ns"), int) and readout["smearing_ns"] >= 0,
            f"Qubit {name!r} needs non-negative integer readout.smearing_ns",
        )
        _require(
            isinstance(transmon.get("thermalization_time_ns"), int)
            and transmon["thermalization_time_ns"] > 0,
            f"Qubit {name!r} needs positive integer transmon.thermalization_time_ns",
        )
        _require(isinstance(operations, dict) and operations, f"Qubit {name!r} must define operations")

        if flux is not None:
            _require(isinstance(flux, dict), f"Qubit {name!r} flux must be an object")
            _require(
                flux.get("flux_point") in FLUX_POINTS,
                f"Qubit {name!r} has invalid flux.flux_point",
            )
            for offset_name in (
                "independent_offset_v",
                "joint_offset_v",
                "min_offset_v",
                "arbitrary_offset_v",
            ):
                _require(
                    isinstance(flux.get(offset_name), (int, float)),
                    f"Qubit {name!r} needs numeric flux.{offset_name}",
                )
            for optional_name in (
                "settle_time_ns",
                "freq_vs_flux_01_quad_term",
                "phi0_current_a",
                "phi0_voltage_v",
            ):
                value = flux.get(optional_name)
                _require(
                    value is None or isinstance(value, (int, float)),
                    f"Qubit {name!r} flux.{optional_name} must be numeric or null",
                )

        for operation_name, pulse_name in operations.items():
            _require(pulse_name in qubit_pulses, f"Qubit {name!r} operation {operation_name!r} references unknown pulse {pulse_name!r}")
            target = qubit_pulses[pulse_name]["target"]
            expected_target = "resonator" if operation_name.startswith("readout") else "qubit"
            _require(target == expected_target, f"Qubit {name!r} operation {operation_name!r} must target {expected_target}")


def _validate_metrics(metrics_document: dict[str, Any], qubits_document: dict[str, Any]) -> None:
    metrics = metrics_document.get("qubits")
    qubits = qubits_document.get("qubits")
    _require(isinstance(metrics, dict), "metrics.json must define qubits")

    for qubit_name in qubits:
        _require(qubit_name in metrics, f"Qubit {qubit_name!r} has no metrics entry")
        qubit_metrics = metrics[qubit_name]
        _require(isinstance(qubit_metrics, dict), f"Metrics for {qubit_name!r} must be an object")
        readout = qubit_metrics.get("readout", {})
        fidelity = readout.get("fidelity_percent", {})
        _require(isinstance(fidelity, dict), f"Metrics for {qubit_name!r} need readout.fidelity_percent")
        for reset_name in ("active", "thermal"):
            value = fidelity.get(reset_name)
            _require(
                value is None or isinstance(value, (int, float)),
                f"Metrics for {qubit_name!r} readout.fidelity_percent.{reset_name} must be numeric or null",
            )


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate a loaded profile, raising ProfileError on the first issue."""
    manifest = profile["manifest"]
    connectivity = profile["connectivity"]
    qubits = profile["qubits"]
    pulses = profile["pulses"]
    metrics = profile.get("metrics")

    for name, document in profile.items():
        _validate_version(document, name)

    _validate_pulses(pulses)
    _validate_qubits(qubits, pulses)
    if metrics is not None:
        _validate_metrics(metrics, qubits)
    _validate_connectivity(connectivity, qubits)

    active_qubits = manifest.get("active_qubits")
    _require(isinstance(active_qubits, list) and active_qubits, "profile.json must define active_qubits")
    for qubit_name in active_qubits:
        _require(qubit_name in qubits["qubits"], f"Active qubit {qubit_name!r} is undefined")


def _load_profile_documents(name: str, root: Path = PROFILES_ROOT) -> dict[str, Any]:
    profile_directory = _profile_directory(root, name)
    manifest = _read_json(profile_directory / "profile.json")
    _validate_version(manifest, "manifest")
    files = manifest.get("files", {})
    return {
        "manifest": manifest,
        "connectivity": _read_json(profile_directory / files.get("connectivity", "connectivity.json")),
        "qubits": _read_json(profile_directory / files.get("qubits", "qubits.json")),
        "pulses": _read_json(profile_directory / files.get("pulses", "pulses.json")),
        **(
            {"metrics": _read_json(profile_directory / files["metrics"])}
            if "metrics" in files
            else {}
        ),
    }


def _selected_hardware(
    connectivity: dict[str, Any], connection: dict[str, Any]
) -> dict[str, Any]:
    """Keep only the controller, FEM, and ports used by one qubit."""
    selected = deepcopy(connectivity)
    selected["connections"] = {}
    selected_controllers: dict[str, Any] = {}

    for direction, reference_name in (
        ("outputs", "xy_output"),
        ("outputs", "resonator_output"),
        ("inputs", "resonator_input"),
    ):
        try:
            reference = connection[reference_name]
            controller_name = reference["controller"]
            fem_name = str(reference["fem"])
            port_name = str(reference["port"])
            source_controller = connectivity["controllers"][controller_name]
            source_fem = source_controller["fems"][fem_name]
            source_port = source_fem[direction][port_name]
        except KeyError as exc:
            raise ProfileError(
                f"Selected qubit has incomplete {reference_name} hardware configuration"
            ) from exc

        controller = selected_controllers.setdefault(
            controller_name,
            {"type": source_controller["type"], "fems": {}},
        )
        fem = controller["fems"].setdefault(
            fem_name,
            {"type": source_fem["type"], "outputs": {}, "inputs": {}},
        )
        fem[direction][port_name] = deepcopy(source_port)

    z_reference = _global_flux_output_reference(connectivity, connection)
    if z_reference is not None:
        try:
            controller_name = z_reference["controller"]
            fem_name = str(z_reference["fem"])
            port_name = str(z_reference["port"])
            source_controller = connectivity["controllers"][controller_name]
            source_fem = source_controller["fems"][fem_name]
            source_port = source_fem["outputs"][port_name]
        except KeyError as exc:
            raise ProfileError(
                "Selected qubit has incomplete z_output hardware configuration"
            ) from exc

        controller = selected_controllers.setdefault(
            controller_name,
            {"type": source_controller["type"], "fems": {}},
        )
        fem = controller["fems"].setdefault(
            fem_name,
            {"type": source_fem["type"], "outputs": {}, "inputs": {}},
        )
        fem["outputs"][port_name] = deepcopy(source_port)

    selected["controllers"] = selected_controllers
    return selected


def _select_qubit(profile: dict[str, Any], qubit: str) -> dict[str, Any]:
    """Return an independently loaded profile containing only one qubit."""
    profile = deepcopy(profile)
    qubits = profile["qubits"]["qubits"]
    _require(
        qubit in qubits,
        f"Qubit {qubit!r} does not exist in profile {profile['manifest']['name']!r}",
    )
    _require(
        qubit in profile["pulses"]["pulses"],
        f"Qubit {qubit!r} has no pulse definitions in profile {profile['manifest']['name']!r}",
    )
    _require(
        qubit in profile["connectivity"]["connections"],
        f"Qubit {qubit!r} has no connectivity entry in profile {profile['manifest']['name']!r}",
    )
    connection = profile["connectivity"]["connections"][qubit]
    lo_frequencies = connection.get("lo_frequencies_hz")
    _require(
        isinstance(lo_frequencies, dict),
        f"Qubit {qubit!r} has no lo_frequencies_hz in profile {profile['manifest']['name']!r}",
    )
    for name in ("xy_output", "resonator_output"):
        _require(
            isinstance(lo_frequencies.get(name), (int, float))
            and lo_frequencies[name] > 0,
            f"Qubit {qubit!r} needs positive lo_frequencies_hz.{name}",
        )
    profile["manifest"] = {
        **profile["manifest"],
        "active_qubits": [qubit],
        "selected_qubit": qubit,
    }
    profile["qubits"]["qubits"] = {qubit: qubits[qubit]}
    profile["pulses"]["pulses"] = {qubit: profile["pulses"]["pulses"][qubit]}
    if "metrics" in profile:
        profile["metrics"]["qubits"] = {qubit: profile["metrics"]["qubits"][qubit]}
    profile["connectivity"] = _selected_hardware(profile["connectivity"], connection)
    profile["connectivity"]["connections"] = {qubit: connection}
    if "global_flux_lines" in profile["connectivity"]:
        referenced_line = connection.get("z_output", {}).get("global_line")
        profile["connectivity"]["global_flux_lines"] = (
            {
                referenced_line: profile["connectivity"]["global_flux_lines"][
                    referenced_line
                ]
            }
            if referenced_line is not None
            else {}
        )
    xy_output = _get_port(
        profile["connectivity"], connection["xy_output"], "outputs", f"{qubit}.xy_output"
    )
    resonator_output = _get_port(
        profile["connectivity"],
        connection["resonator_output"],
        "outputs",
        f"{qubit}.resonator_output",
    )
    xy_output["lo_frequency_hz"] = lo_frequencies["xy_output"]
    resonator_output["lo_frequency_hz"] = lo_frequencies["resonator_output"]
    validate_profile(profile)
    return profile


class Profile:
    """Repository-backed device profile with explicit load/save behavior."""

    def __init__(
        self,
        name: str = "main",
        *,
        qubit: str | None = None,
        root: Path | str = PROFILES_ROOT,
    ) -> None:
        self.name = name
        self.qubit = qubit
        self.root = Path(root)
        self.documents: dict[str, Any] | None = None

    @property
    def directory(self) -> Path:
        return _profile_directory(self.root, self.name)

    def for_qubit(self, qubit: str) -> "Profile":
        return Profile(self.name, qubit=qubit, root=self.root)

    def load(self, *, qubit: str | None = None) -> dict[str, Any]:
        selected_qubit = self.qubit if qubit is None else qubit
        profile = _load_profile_documents(self.name, self.root)
        validate_profile(profile)
        self.documents = deepcopy(profile)

        if selected_qubit is None:
            return profile
        if profile["manifest"].get("build_mode") != "single_qubit":
            raise ProfileError(
                f"Profile {self.name!r} does not support selecting a single qubit"
            )
        return _select_qubit(profile, selected_qubit)

    def save(self, documents: dict[str, Any] | None = None) -> None:
        profile = deepcopy(documents if documents is not None else self.documents)
        if profile is None:
            profile = _load_profile_documents(self.name, self.root)
        if profile["manifest"].get("selected_qubit") is not None:
            raise ProfileError(
                "Refusing to save a selected single-qubit projection over the full profile"
            )
        validate_profile(profile)

        files = profile["manifest"].get("files", {})
        paths = {
            "manifest": self.directory / "profile.json",
            "connectivity": self.directory / files.get("connectivity", "connectivity.json"),
            "qubits": self.directory / files.get("qubits", "qubits.json"),
            "pulses": self.directory / files.get("pulses", "pulses.json"),
        }
        if "metrics" in profile:
            paths["metrics"] = self.directory / files.get("metrics", "metrics.json")
        for key, path in paths.items():
            _write_json(path, profile[key])
        self.documents = deepcopy(profile)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        suffix = f", qubit={self.qubit!r}" if self.qubit is not None else ""
        return f"Profile({self.name!r}{suffix})"


def set_active_profile(profile: Profile | str) -> Profile:
    global _active_profile
    _active_profile = profile if isinstance(profile, Profile) else Profile(str(profile))
    return _active_profile


def clear_active_profile() -> None:
    global _active_profile
    _active_profile = None


def current_profile() -> Profile:
    if _active_profile is not None:
        return _active_profile
    return Profile(os.environ.get("QUAM_PROFILE", "main"))


def current_profile_name() -> str:
    return current_profile().name


def load_profile(name: str = "main", *, qubit: str | None = None) -> dict[str, Any]:
    """Load and validate a named profile from the profiles directory."""
    return Profile(name, qubit=qubit).load()




if __name__ == "__main__":
    try:
        profile = load_profile()
        print("Profile loaded and validated successfully.")
    except ProfileError as exc:
        print(f"Profile validation error: {exc}")
