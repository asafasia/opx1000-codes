"""Make a standalone Figure 4 measurement overlay without updating the paper.

Reuse the original generator's plotting statements with its exported figure
arrays. Never run its acquisition-data loader or paper-output writers. Only
the requested preview directory receives output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--exclude-duration-us", type=float, action="append", default=[],
                        help="Omit a simulation duration from the preview curves and legend.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Preview completed cutoffs from an ongoing scan, labeled as partial.")
    args = parser.parse_args()
    paper = args.paper_root.resolve()
    run = args.run_directory.resolve()
    output = args.output_directory.resolve()
    if output == paper or paper in output.parents:
        raise ValueError("Preview output must be outside the paper project.")
    settings_path, results_path = run / "settings.json", run / "summary.json"
    settings_bytes, results_bytes = settings_path.read_bytes(), results_path.read_bytes()
    settings = json.loads(settings_bytes)
    summary = json.loads(results_bytes)
    records = summary["records"]
    if settings["amplitude_points"] != 1 or not records:
        raise ValueError("Require a single-amplitude scan with at least one completed cutoff.")
    if not summary["complete"] and not args.allow_partial:
        raise ValueError("Scan is incomplete; use --allow-partial for a labeled progress preview.")
    saved_figure = paper / "figures/paper/04_duration_cutoff_scan.npz"
    with np.load(saved_figure) as original:
        saved_arrays = {k: original[k].copy() for k in original.files}
        reference_hz = float(original["reference_linewidth_hz"])
        target_amplitude = float(original["slice_amplitude_mhz"])
        target_duration = float(original["duration_us"])
    records = sorted(records, key=lambda r: r["cutoff"])
    measured_amplitude = float(records[0]["nominal_rabi_mhz"])
    measured_duration = float(settings["pulse_length_ns"]) / 1000
    np.testing.assert_allclose([r["nominal_rabi_mhz"] for r in records], measured_amplitude)
    np.testing.assert_allclose([r["pulse_length_ns"] / 1000 for r in records], measured_duration)
    if measured_duration not in saved_arrays["slice_durations_us"]:
        raise ValueError("Measured duration is not represented among the figure's cutoff slices.")
    source = paper / "scripts/make_main_duration_cutoff_scan.py"
    protected = [source, *list((paper / "figures/paper").glob("04_duration_cutoff_scan*")),
                 paper / "paper/main.tex", paper / "paper/main.pdf",
                 paper / "paper/figures/04_duration_cutoff_scan_float.tex"]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in protected if p.is_file()}
    sys.path.insert(0, str(paper))
    spec = importlib.util.spec_from_file_location("paper_figure4_preview_source", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The exported NPZ is the exact data used by the published figure. Some
    # upstream simulation folders may have been archived; they are not needed.
    main_node = next(n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main_node.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == "apply_figure_style")
    end = next(i for i, n in enumerate(main_node.body)
               if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == "save_figure")
    if end <= start:
        raise RuntimeError("Original generator's plotting section was not found.")
    plot_code = ast.Module(body=main_node.body[start:end], type_ignores=[])
    scope = vars(module).copy()
    original_provenance = json.loads(saved_figure.with_name("04_duration_cutoff_scan_provenance.json").read_text())
    durations = saved_arrays["slice_durations_us"]
    keep = ~np.isin(durations, args.exclude_duration_us)
    if not np.any(keep) or not np.any(durations[keep] == target_duration):
        raise ValueError("Keep the map duration in the simulation slices.")
    scope.update(
        cutoffs=saved_arrays["cutoffs"], amps=saved_arrays["rabi_mhz"],
        lengths=durations[keep],
        COLORS=[module.COLORS[i] for i in np.flatnonzero(keep)],
        LINESTYLES=[module.LINESTYLES[i] for i in np.flatnonzero(keep)],
        slice_index=int(saved_arrays["slice_amplitude_index"]),
        accepted=saved_arrays["accepted"],
        data={k: saved_arrays[k] for k in ("accepted", "fwhm_hz", "contrast", "contrast_above_threshold")},
        slices={k: saved_arrays[k][keep] for k in ("slice_accepted", "slice_fwhm_hz", "slice_contrast")},
        config=original_provenance["source_configuration"],
    )
    exec(compile(plot_code, str(source), "exec"), scope)
    np.testing.assert_allclose(scope["reference_hz"], reference_hz)
    fig = scope["fig"]
    # Identify panels from their original panel labels, not implicit axes order.
    panels = {text.get_text(): ax for ax in fig.axes for text in ax.texts
              if text.get_text() in ("(c)", "(d)")}
    if set(panels) != {"(c)", "(d)"}:
        raise RuntimeError("Could not locate Figure 4 cutoff-slice panels.")
    x = np.array([r["cutoff"] for r in records])
    valid = np.array([r["accepted"] for r in records])
    width = np.array([r["fwhm_hz"] if r["fwhm_hz"] is not None else np.nan for r in records])
    contrast = np.array([r["contrast"] if r["contrast"] is not None else np.nan for r in records])
    marker = dict(marker="D", markersize=2.0 if len(records) > 30 else 3.2, linestyle="none", color="black",
                  markeredgecolor="white", markeredgewidth=.45, zorder=10, clip_on=False)
    panels["(c)"].plot(x[valid], width[valid]/reference_hz, **marker)
    panels["(d)"].plot(x[np.isfinite(contrast)], contrast[np.isfinite(contrast)], **marker)
    # Add space below the original simulation legend while retaining panel sizes.
    fw, fh = fig.get_size_inches()
    extra = .62
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0, (p.y0*fh+extra)/(fh+extra), p.width, p.height*fh/(fh+extra)])
    fig.set_size_inches(fw, fh+extra)
    for legend in fig.legends:
        legend.set_bbox_to_anchor((.55, extra/(fh+extra)))
    experiment_marker = Line2D([], [], **{k: v for k, v in marker.items() if k != "zorder"})
    experiment_power = settings["first_pulse_parameters"]["edge_smoothing_power"]
    simulation_power = original_provenance["source_configuration"]["power"]
    simulation_beta = original_provenance["source_configuration"]["drag_beta"]
    label = (rf"q1: ${measured_duration:g}\,\mu$s, $p={experiment_power}$, $\beta_{{\rm lab}}={settings['drag_beta']:.5f}$"
             + "\n" + rf"$\Omega_0/2\pi={measured_amplitude:.3f}$ MHz; {settings['num_shots']:,} averages")
    fig.legend([experiment_marker], [label], loc="lower center", bbox_to_anchor=(.54,.055),
               frameon=False, fontsize=5.5, handlelength=1, handletextpad=.45)
    progress = f"{'Complete' if summary['complete'] else 'Partial'}: {len(records)}/{settings['cutoff_points']} cutoffs"
    comparison = (f"Simulation: p={simulation_power:g}, beta={simulation_beta:g}, slice {target_amplitude:.3f} MHz"
                  + "\n" + f"{progress}; common linewidth reference {reference_hz/1000:.3f} kHz")
    fig.text(.54, .013, comparison,
             ha="center", fontsize=4.3, color="0.3")
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "figure4_with_q1_cutoff_measurements_preview"
    for extension in ("png", "svg"):
        fig.savefig(stem.with_suffix("."+extension), dpi=350)
    plt.close(fig)
    for path, original_hash in before.items():
        if digest(Path(path)) != original_hash:
            raise RuntimeError(f"Original paper source changed during preview generation: {path}")
    provenance = dict(
        included_in_manuscript=False, experiment_directory=str(run),
        acquisition_complete=summary["complete"], measured_cutoffs=len(records),
        accepted_widths=int(valid.sum()),
        experiment_settings=settings, experiment_records=records,
        reference_linewidth_hz=reference_hz, experiment_overlay_panels=["c", "d"],
        simulation="Original Figure 4 source data and curves; no recomputation or interpolation",
        comparison_note=comparison + ". Beta conventions differ. Two-state experimental readout does not separately resolve Pe and Pf.",
        measured_duration_us=measured_duration, simulated_map_duration_us=target_duration,
        displayed_simulation_durations_us=durations[keep].tolist(),
        excluded_simulation_durations_us=durations[~keep].tolist(),
        measured_nominal_rabi_mhz=measured_amplitude, simulated_slice_rabi_mhz=target_amplitude,
        experiment_edge_power=experiment_power, simulation_edge_power=simulation_power,
        error_bars="None: saved half-depth metrics do not provide confidence intervals",
        paper_files_unchanged=True, paper_sha256=before,
        experiment_sha256={str(settings_path): hashlib.sha256(settings_bytes).hexdigest(),
                           str(results_path): hashlib.sha256(results_bytes).hexdigest()},
    )
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(stem.with_suffix(".png"))
    print(f"Validated {len(records)} points; normalization = {reference_hz/1000:.4f} kHz.")
    print(f"Verified {len(before)} original paper/figure files unchanged.")


if __name__ == "__main__":
    main()
