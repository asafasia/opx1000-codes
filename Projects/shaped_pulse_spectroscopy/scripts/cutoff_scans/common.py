"""Shared helpers for cutoff-region scan scripts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cutoff_regions.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "echo_lorentzian_cutoff_sweep"
for path in (PROJECT_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrations_v2.base import CalibrationOptions
from experiments.cutoff_optimization import (
    plot_cutoff_summary,
    plot_fwhm_heatmap,
    plot_per_cutoff_traces,
    run_cutoff_sweep,
)
from shaped_pulse_spectroscopy.parameters import Parameters
from quam_config import create_machine


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--pulse-length-ns", type=int)
    parser.add_argument("--template-length-ns", type=int)
    parser.add_argument("--peak-amplitude", type=float)
    parser.add_argument("--min-amp-factor", type=float)
    parser.add_argument("--max-amp-factor", type=float)
    parser.add_argument("--amp-factor-step", type=float)
    parser.add_argument("--echo", action=argparse.BooleanOptionalAction)


def region_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--region", required=True)
    parser.add_argument("--domains", nargs="*")
    add_common_arguments(parser)
    return parser


def domain_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--region", required=True)
    parser.add_argument("--domain", required=True)
    add_common_arguments(parser)
    return parser


def run_region_from_args(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    region = config[args.region]
    domains = region["domains"]
    if args.domains:
        wanted = set(args.domains)
        domains = [domain for domain in domains if domain["name"] in wanted]
    if not domains:
        raise SystemExit("No matching domains selected.")

    campaign_dir = campaign_output_dir(args.output_root, args.region, args.campaign_id)
    for domain in domains:
        run_domain_config(args, region, domain, campaign_dir)
    return 0


def run_domain_from_args(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    region = config[args.region]
    domain = next(
        (candidate for candidate in region["domains"] if candidate["name"] == args.domain),
        None,
    )
    if domain is None:
        raise SystemExit(f"Unknown domain {args.domain!r} for region {args.region!r}.")
    campaign_dir = campaign_output_dir(args.output_root, args.region, args.campaign_id)
    run_domain_config(args, region, domain, campaign_dir)
    return 0


def campaign_output_dir(output_root: Path, region_name: str, campaign_id: str | None) -> Path:
    campaign = campaign_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / region_name / campaign


def run_domain_config(
    args: argparse.Namespace,
    region: dict[str, Any],
    domain: dict[str, Any],
    campaign_dir: Path,
) -> None:
    output_dir = campaign_dir / str(domain["name"])
    parameters = build_parameters(args, region, domain)
    cutoffs = [float(cutoff) for cutoff in region["cutoffs"]]
    print_plan(args, region, domain, output_dir, cutoffs, parameters)
    if not args.execute:
        return

    run_cutoff_sweep(
        parameters,
        machine=create_machine(qubit=args.qubit),
        qubit=args.qubit,
        cutoffs=cutoffs,
        output_dir=output_dir,
        options=CalibrationOptions(),
        auto_connect=not args.simulate,
    )


def build_parameters(
    args: argparse.Namespace,
    region: dict[str, Any],
    domain: dict[str, Any],
) -> Parameters:
    defaults = dict(region.get("defaults", {}))
    overrides = {
        "num_shots": args.num_shots,
        "pulse_length_ns": args.pulse_length_ns,
        "template_length_ns": args.template_length_ns,
        "peak_amplitude": args.peak_amplitude,
        "min_amp_factor": args.min_amp_factor,
        "max_amp_factor": args.max_amp_factor,
        "amp_factor_step": args.amp_factor_step,
        "echo": args.echo,
    }
    defaults.update({key: value for key, value in overrides.items() if value is not None})

    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.pulse_shape = "root_lorentzian"
    parameters.echo = bool(defaults.get("echo", True))
    parameters.num_shots = int(defaults.get("num_shots", 60))
    parameters.lorentzian_length_in_ns = int(defaults.get("pulse_length_ns", 20000))
    parameters.waveform_template_length_in_ns = int(
        defaults.get("template_length_ns", parameters.lorentzian_length_in_ns)
    )
    parameters.lorentzian_peak_amplitude = float(defaults.get("peak_amplitude", 0.2))
    parameters.min_amp_factor = float(defaults.get("min_amp_factor", 0.0))
    parameters.max_amp_factor = float(defaults.get("max_amp_factor", 1.0))
    parameters.amp_factor_step = float(defaults.get("amp_factor_step", 0.04))
    parameters.frequency_span_in_mhz = float(domain["span_mhz"])
    parameters.frequency_step_in_mhz = float(domain["step_mhz"])
    parameters.simulate = bool(args.simulate)
    return parameters


def print_plan(
    args: argparse.Namespace,
    region: dict[str, Any],
    domain: dict[str, Any],
    output_dir: Path,
    cutoffs: list[float],
    parameters: Parameters,
) -> None:
    mode = "EXECUTE" if args.execute else "PLAN ONLY"
    print(f"[{mode}] {args.region} / {domain['name']}")
    print(f"  description: {region.get('description', '')}")
    print(f"  qubit: {args.qubit}")
    print(f"  cutoffs: {cutoffs}")
    print(f"  detuning span/step MHz: {parameters.frequency_span_in_mhz:g} / {parameters.frequency_step_in_mhz:g}")
    print(f"  shots: {parameters.num_shots}")
    print(f"  pulse/template ns: {parameters.lorentzian_length_in_ns} / {parameters.waveform_template_length_in_ns}")
    print(f"  output: {output_dir}")
    if not args.execute:
        print("  add --execute to run this hardware/simulation scan")


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def coerce_record_numbers(record: dict[str, Any]) -> dict[str, Any]:
    converted = dict(record)
    for key, value in list(converted.items()):
        if value in {"", "None", None}:
            converted[key] = float("nan")
            continue
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def collect_domain_records(region_dir: Path, filename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for csv_path in sorted(region_dir.glob(f"domain_*/{filename}")):
        domain = csv_path.parent.name
        for record in read_records(csv_path):
            record = coerce_record_numbers(record)
            record["domain"] = domain
            records.append(record)
    return records


def build_region_summary(region_dir: Path) -> None:
    fit_records = collect_domain_records(region_dir, "cutoff_sweep_fit_results.csv")
    best_records = collect_domain_records(region_dir, "cutoff_sweep_best_signal.csv")
    write_records(region_dir / "region_fit_results.csv", fit_records)
    write_records(region_dir / "region_best_signal.csv", best_records)


def replot_region(region_dir: Path) -> None:
    fit_records = read_records(region_dir / "region_fit_results.csv")
    best_records = read_records(region_dir / "region_best_signal.csv")
    if not fit_records:
        fit_records = collect_domain_records(region_dir, "cutoff_sweep_fit_results.csv")
    if not best_records:
        best_records = collect_domain_records(region_dir, "cutoff_sweep_best_signal.csv")
    fit_records = [coerce_record_numbers(record) for record in fit_records]
    best_records = [coerce_record_numbers(record) for record in best_records]
    plot_cutoff_summary(best_records, region_dir)
    plot_fwhm_heatmap(fit_records, region_dir)
    plot_per_cutoff_traces(fit_records, region_dir)
