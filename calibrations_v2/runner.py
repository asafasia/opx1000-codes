"""Command-line wrapper around class-based calibrations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .registry import CALIBRATIONS, CalibrationEntry, get_entry


@dataclass
class RunnerCalibrationOptions:
    """Runtime switches matching calibrations_v2.core.CalibrationOptions."""

    save_raw_data: bool = True
    save_analysis_result: bool = True
    save_figures: bool = True
    analyse_data: bool = True
    plot_data: bool = True
    update_state: bool = True
    propose_profile_update: bool = True
    apply_profile_update: bool = False


OPTION_FIELDS = set(RunnerCalibrationOptions.__dataclass_fields__)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def coerce_value(text: str) -> Any:
    """Coerce CLI strings into simple Python values."""
    value = text.strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected name=value, got {raw!r}")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Expected name=value, got {raw!r}")
    return name, coerce_value(value)


def parameter_fields(parameters_class: type[Any]) -> dict[str, Any]:
    """Return annotated parameter fields across the class hierarchy."""
    fields: dict[str, Any] = {}
    for cls in reversed(parameters_class.mro()):
        annotations = getattr(cls, "__annotations__", {})
        fields.update(annotations)
    return fields


def public_parameter_values(parameters: Any) -> dict[str, Any]:
    fields = parameter_fields(type(parameters))
    values = {}
    for name in fields:
        if name.startswith("_"):
            continue
        values[name] = getattr(parameters, name, None)
    return values


def build_parameters(
    parameters_class: type[Any],
    assignments: Iterable[tuple[str, Any]] = (),
    *,
    simulate: bool = False,
    load_data_id: str | None = None,
) -> Any:
    parameters = parameters_class()
    if simulate:
        setattr(parameters, "simulate", True)
    if load_data_id is not None:
        setattr(parameters, "load_data_id", load_data_id)
    valid_fields = parameter_fields(parameters_class)
    for name, value in assignments:
        if valid_fields and name not in valid_fields and not hasattr(parameters, name):
            valid = ", ".join(sorted(valid_fields))
            raise ValueError(f"Unknown parameter {name!r}. Known parameters: {valid}")
        setattr(parameters, name, value)
    return parameters


def fallback_parameters_class(assignments: Iterable[tuple[str, Any]]) -> type[Any]:
    annotations = {
        "simulate": bool,
        "load_data_id": str | None,
        **{name: type(value) for name, value in assignments},
    }

    class FallbackParameters(SimpleNamespace):
        pass

    FallbackParameters.__annotations__ = annotations
    FallbackParameters.simulate = False
    FallbackParameters.load_data_id = None
    return FallbackParameters


def build_options(
    assignments: Iterable[tuple[str, Any]] = (),
    *,
    apply_profile_update: bool = False,
    save_raw_data: bool = True,
    save_figures: bool = True,
    plot_data: bool = True,
) -> RunnerCalibrationOptions:
    options = RunnerCalibrationOptions(
        apply_profile_update=apply_profile_update,
        save_raw_data=save_raw_data,
        save_figures=save_figures,
        plot_data=plot_data,
    )
    for name, value in assignments:
        if name not in OPTION_FIELDS:
            valid = ", ".join(sorted(OPTION_FIELDS))
            raise ValueError(f"Unknown option {name!r}. Known options: {valid}")
        setattr(options, name, bool(value) if isinstance(getattr(options, name), bool) else value)
    return options


def load_recipe(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_assignments(payload: Mapping[str, Any], section: str) -> list[tuple[str, Any]]:
    values = payload.get(section, {})
    if not isinstance(values, Mapping):
        raise ValueError(f"Recipe section {section!r} must be an object.")
    return [(str(name), value) for name, value in values.items()]


def print_entry_list() -> None:
    for key in sorted(CALIBRATIONS):
        entry = CALIBRATIONS[key]
        suffix = f" - {entry.description}" if entry.description else ""
        print(f"{key}: {entry.module}.{entry.class_name}{suffix}")


def describe_entry(entry: CalibrationEntry) -> None:
    try:
        parameters_class = entry.load_parameters_class()
    except Exception as error:
        print(f"name: {entry.key}")
        print(f"class: {entry.module}.{entry.class_name}")
        print(f"parameter_import_error: {error}")
        return
    print(f"name: {entry.key}")
    print(f"class: {entry.module}.{entry.class_name}")
    if entry.description:
        print(f"description: {entry.description}")
    print("parameters:")
    defaults = public_parameter_values(parameters_class())
    for name, annotation in parameter_fields(parameters_class).items():
        if name.startswith("_"):
            continue
        default = defaults.get(name)
        print(f"  {name}: {default!r} ({annotation})")


def run_entry(
    entry: CalibrationEntry,
    *,
    parameter_assignments: Iterable[tuple[str, Any]],
    option_assignments: Iterable[tuple[str, Any]],
    profile_name: str | None,
    qubit: str | None,
    simulate: bool,
    load_data_id: str | None,
    apply: bool,
    auto_connect: bool,
    dry_run: bool,
    no_save: bool,
    no_plot: bool,
) -> int:
    parameter_import_error = None
    try:
        parameters_class = entry.load_parameters_class()
    except Exception as error:
        if not dry_run:
            raise
        parameter_import_error = error
        parameters_class = fallback_parameters_class(parameter_assignments)
    parameters = build_parameters(
        parameters_class,
        parameter_assignments,
        simulate=simulate,
        load_data_id=load_data_id,
    )
    options = build_options(
        option_assignments,
        apply_profile_update=apply,
        save_raw_data=not no_save,
        save_figures=not no_save,
        plot_data=not no_plot,
    )

    if dry_run:
        print(
            json.dumps(
                {
                    "calibration": entry.key,
                    "class": f"{entry.module}.{entry.class_name}",
                    "profile_name": profile_name,
                    "qubit": qubit,
                    "parameter_import_error": (
                        str(parameter_import_error) if parameter_import_error else None
                    ),
                    "parameters": public_parameter_values(parameters),
                    "options": vars(options),
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    calibration_class = entry.load_class()
    calibration = calibration_class(
        parameters=parameters,
        profile_name=profile_name,
        qubit=qubit,
        options=options,
        auto_connect=auto_connect,
    )
    status = calibration.run()
    summary = {
        "name": status.name,
        "mode": status.mode,
        "outcomes": dict(status.outcomes),
        "raw_data_saved": status.raw_data_saved,
        "figures_saved": status.figures_saved,
        "profile_update_proposed": status.profile_update_proposed,
        "run_directory": calibration.namespace.get("calibration_run_directory"),
        "profile_update_proposal": calibration.namespace.get("profile_update_proposal"),
    }
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run class-based calibrations from the terminal.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered calibrations.")

    describe = subparsers.add_parser("describe", help="Show parameters for one calibration.")
    describe.add_argument("calibration", help="Friendly calibration name, for example resonator.")

    run = subparsers.add_parser("run", help="Run one calibration.")
    run.add_argument("calibration", nargs="?", help="Friendly calibration name.")
    run.add_argument("--recipe", type=Path, help="JSON recipe with calibration, parameters, and options.")
    run.add_argument("--set", dest="sets", action="append", type=parse_assignment, default=[], help="Parameter override as name=value.")
    run.add_argument("--option", dest="options", action="append", type=parse_assignment, default=[], help="CalibrationOptions override as name=value.")
    run.add_argument("--profile", dest="profile_name", help="Profile name passed to create_machine.")
    run.add_argument("--qubit", help="Qubit name passed to create_machine.")
    run.add_argument("--simulate", action="store_true", help="Simulate instead of executing.")
    run.add_argument("--load", dest="load_data_id", help="Load a saved calibration run directory.")
    run.add_argument("--apply", action="store_true", help="Apply staged profile updates after confirmation.")
    run.add_argument("--auto-connect", action="store_true", help="Connect and close existing QMs during construction.")
    run.add_argument("--dry-run", action="store_true", help="Print resolved configuration without creating the machine.")
    run.add_argument("--no-save", action="store_true", help="Disable raw-data and figure saves.")
    run.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print_entry_list()
        return 0
    if args.command == "describe":
        describe_entry(get_entry(args.calibration))
        return 0

    recipe = load_recipe(args.recipe) if args.recipe else {}
    calibration_name = args.calibration or recipe.get("calibration")
    if not calibration_name:
        raise SystemExit("run requires a calibration name or a recipe with 'calibration'.")
    parameter_assignments = recipe_assignments(recipe, "parameters") + list(args.sets)
    option_assignments = recipe_assignments(recipe, "options") + list(args.options)
    profile_name = args.profile_name if args.profile_name is not None else recipe.get("profile")
    qubit = args.qubit if args.qubit is not None else recipe.get("qubit")
    return run_entry(
        get_entry(str(calibration_name)),
        parameter_assignments=parameter_assignments,
        option_assignments=option_assignments,
        profile_name=profile_name,
        qubit=qubit,
        simulate=bool(args.simulate or recipe.get("simulate", False)),
        load_data_id=args.load_data_id or recipe.get("load"),
        apply=bool(args.apply or recipe.get("apply", False)),
        auto_connect=bool(args.auto_connect or recipe.get("auto_connect", False)),
        dry_run=bool(args.dry_run),
        no_save=bool(args.no_save or recipe.get("no_save", False)),
        no_plot=bool(args.no_plot or recipe.get("no_plot", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
