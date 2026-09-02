"""Build a self-contained HTML report for a pulse-length spectroscopy campaign."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
from string import Template
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <campaign_dir>/pulse_length_report.html)",
    )
    return parser.parse_args()


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def finite_float(value: str | float | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def acquisition_seconds(run: dict[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(run["started_at"])
        finished = datetime.fromisoformat(run["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return (finished - started).total_seconds()


def build_length_rows(
    runs: list[dict[str, Any]], comparison_rows: list[dict[str, str]]
) -> str:
    rows = []
    for run in runs:
        length = float(run["pulse_length_us"])
        selected = [
            row
            for row in comparison_rows
            if finite_float(row.get("pulse_length_us")) == length
        ]
        paired = [
            row
            for row in selected
            if finite_float(row.get("measured_fwhm_t2_units")) is not None
            and finite_float(row.get("simulated_fwhm_t2_units")) is not None
        ]
        measured = [
            value
            for row in paired
            if (value := finite_float(row.get("measured_fwhm_t2_units"))) is not None
        ]
        simulated = [
            value
            for row in paired
            if (value := finite_float(row.get("simulated_fwhm_t2_units"))) is not None
        ]
        duration = acquisition_seconds(run)
        rows.append(
            "<tr>"
            f"<td>{length:g} µs</td>"
            f"<td><span class='status'>{run.get('status', 'unknown')}</span></td>"
            f"<td>{format_number(duration, 1)} s</td>"
            f"<td>{len(paired)} / {len(selected)}</td>"
            f"<td>{format_number(median(measured) if measured else None)}</td>"
            f"<td>{format_number(median(simulated) if simulated else None)}</td>"
            "</tr>"
        )
    return "".join(rows)


REPORT_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pulse-length spectroscopy report · $qubit</title>
<style>
:root { --ink:#17212b; --muted:#62707e; --paper:#f6f3ed; --card:#fffdf8;
  --line:#ded8cc; --blue:#176b87; --orange:#d76025; --green:#247659; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.5 Inter,Segoe UI,Arial,sans-serif; }
header { color:white; padding:54px max(24px,calc((100vw - 1220px)/2));
  background:linear-gradient(125deg,#102c3a 0%,#176b87 58%,#439a8c 100%); }
header h1 { margin:0 0 8px; font:700 clamp(30px,5vw,54px)/1.05 Georgia,serif; letter-spacing:-.025em; }
header p { margin:0; opacity:.84; font-size:17px; }
.pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:24px; }
.pill { padding:7px 11px; border:1px solid #ffffff4d; border-radius:999px; background:#ffffff12; }
nav { position:sticky; top:0; z-index:20; display:flex; gap:6px; padding:10px max(20px,calc((100vw - 1220px)/2));
  background:#fffdf8e8; backdrop-filter:blur(10px); border-bottom:1px solid var(--line); }
nav a { color:var(--ink); text-decoration:none; font-weight:650; padding:7px 12px; border-radius:8px; }
nav a:hover { background:#e8f2f3; color:var(--blue); }
main { max-width:1220px; margin:auto; padding:30px 20px 80px; }
section { scroll-margin-top:72px; margin:0 0 38px; }
h2 { margin:0 0 8px; font:700 30px/1.15 Georgia,serif; }
h3 { margin:28px 0 8px; font:700 23px/1.2 Georgia,serif; }
.lead { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:0 10px 30px #2d3d4410; overflow:hidden; }
.sweep-head { display:flex; justify-content:space-between; gap:16px; align-items:center; padding:16px 18px; border-bottom:1px solid var(--line); }
.sweep-title { font-size:22px; font-weight:750; }
.counter { color:var(--muted); }
.viewer-pair { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
.sweep-panel { min-width:0; background:#fff; }
.sweep-panel h3 { margin:0; padding:10px 14px; color:var(--muted); font:700 13px/1.2 Inter,Segoe UI,Arial,sans-serif; text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid var(--line); }
.viewer { min-height:420px; display:grid; align-items:start; padding:10px; background:#fff; }
.viewer img { display:block; width:100%; max-width:100%; height:auto; object-fit:contain; }
.controls { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:14px; padding:14px 18px; border-top:1px solid var(--line); }
button { border:1px solid var(--line); background:white; color:var(--ink); border-radius:9px; padding:9px 14px; font-weight:700; cursor:pointer; }
button:hover { border-color:var(--blue); color:var(--blue); }
input[type=range] { width:100%; accent-color:var(--blue); }
.thumbs { display:grid; grid-template-columns:repeat(10,minmax(64px,1fr)); gap:8px; margin-top:10px; }
.thumb { padding:8px 4px; color:var(--muted); }
.thumb.active { background:var(--blue); color:#fff; border-color:var(--blue); }
.analysis-grid { display:grid; grid-template-columns:1.35fr .65fr; gap:18px; }
.figure-card img { width:100%; display:block; background:white; }
.caption { padding:12px 15px; color:var(--muted); border-top:1px solid var(--line); }
.metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0; }
.metric { padding:16px; }
.metric strong { display:block; color:var(--blue); font-size:24px; }
.metric span { color:var(--muted); }
.table-wrap { overflow:auto; }
table { width:100%; border-collapse:collapse; }
th,td { padding:11px 13px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.status { color:var(--green); font-weight:750; }
@media (max-width:850px) { .viewer-pair,.analysis-grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.thumbs{grid-template-columns:repeat(5,1fr)} }
@media (max-width:520px) { .controls{grid-template-columns:1fr 1fr}.controls input{grid-column:1/-1;grid-row:1}.metrics{grid-template-columns:1fr}.viewer{min-height:260px} }
</style>
</head>
<body>
<header>
  <h1>Pulse-length spectroscopy</h1>
  <p>Echo shaped-pulse detuning × amplitude campaign · $created_at</p>
  <div class="pills">
    <span class="pill">$qubit</span><span class="pill">cutoff $cutoff</span><span class="pill">$length_count pulse lengths</span>
    <span class="pill">$frequency_points × $amplitude_points points</span><span class="pill">$frequency_span_mhz MHz span</span>
    <span class="pill">0.01–1 V log sweep</span><span class="pill">AC Stark disabled</span>
  </div>
</header>
<nav><a href="#sweeps">Length sweeps</a><a href="#length-fwhm">FWHM vs length</a><a href="#analysis">Post-analysis</a></nav>
<main>
  <section id="sweeps">
    <h2>Length sweep explorer</h2>
    <p class="lead">Use the slider, buttons, thumbnails, or keyboard arrows to compare each measured map with its matched qutrit simulation.</p>
    <div class="card">
      <div class="sweep-head"><div id="sweepTitle" class="sweep-title"></div><div id="counter" class="counter"></div></div>
      <div class="viewer-pair">
        <div class="sweep-panel"><h3>Experiment</h3><div class="viewer"><img id="experimentImage" alt="Measured pulse-length spectroscopy sweep"></div></div>
        <div class="sweep-panel" id="simulationPanel"><h3>Matched simulation</h3><div class="viewer"><img id="simulationImage" alt="Simulated pulse-length spectroscopy sweep"></div></div>
      </div>
      <div class="controls"><button id="previous" type="button">← Previous</button><input id="lengthSlider" type="range"><button id="next" type="button">Next →</button></div>
    </div>
    <div id="thumbnails" class="thumbs"></div>
  </section>

  <section id="analysis">
    <h2>FWHM post-analysis</h2>
    <p class="lead">Experiment and matched qutrit simulation in Ramsey linewidth units. The dashed horizontal line at 1 marks the $t2_limit_khz kHz T₂* limit.</p>
    <div class="metrics">
      <div class="card metric"><strong>$t2_star_us µs</strong><span>Ramsey T₂*</span></div>
      <div class="card metric"><strong>$t2_limit_khz kHz</strong><span>1 / (πT₂*)</span></div>
      <div class="card metric"><strong>$finite_comparisons</strong><span>finite comparisons</span></div>
      <div class="card metric"><strong>$comparison_points</strong><span>total amplitude points</span></div>
    </div>
    <div class="analysis-grid">
      <div class="card figure-card"><img src="$overlay_image" alt="FWHM by pulse length"><div class="caption">Normalized FWHM across all pulse lengths and amplitudes.</div></div>
      <div class="card figure-card"><img src="$parity_image" alt="Measured versus simulated FWHM"><div class="caption">Measured-versus-simulated parity; dashed diagonal is 1:1.</div></div>
    </div>
    $rms_analysis_section
  </section>

</main>
<script>
const sweeps = $sweeps_json;
let selected = 0;
const experimentImage = document.getElementById('experimentImage');
const simulationImage = document.getElementById('simulationImage');
const simulationPanel = document.getElementById('simulationPanel');
const title = document.getElementById('sweepTitle');
const counter = document.getElementById('counter');
const slider = document.getElementById('lengthSlider');
const thumbnailBox = document.getElementById('thumbnails');
slider.min = 0; slider.max = sweeps.length - 1; slider.step = 1;
sweeps.forEach((item, index) => {
  const button = document.createElement('button');
  button.className = 'thumb'; button.textContent = item.length + ' µs';
  button.addEventListener('click', () => showSweep(index));
  thumbnailBox.appendChild(button);
});
function showSweep(index) {
  selected = (index + sweeps.length) % sweeps.length;
  const item = sweeps[selected];
  experimentImage.src = item.experimentSrc;
  experimentImage.alt = item.length + ' µs measured echo spectroscopy sweep';
  if (item.simulationSrc) {
    simulationImage.src = item.simulationSrc;
    simulationImage.alt = item.length + ' µs matched qutrit simulation';
    simulationPanel.hidden = false;
  } else {
    simulationImage.removeAttribute('src');
    simulationPanel.hidden = true;
  }
  title.textContent = item.length + ' µs echo sweep · cutoff ' + item.cutoff;
  counter.textContent = (selected + 1) + ' of ' + sweeps.length;
  slider.value = selected;
  [...thumbnailBox.children].forEach((button, i) => button.classList.toggle('active', i === selected));
}
slider.addEventListener('input', event => showSweep(Number(event.target.value)));
document.getElementById('previous').addEventListener('click', () => showSweep(selected - 1));
document.getElementById('next').addEventListener('click', () => showSweep(selected + 1));
document.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') showSweep(selected - 1);
  if (event.key === 'ArrowRight') showSweep(selected + 1);
});
showSweep(0);
</script>
</body>
</html>"""
)


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    analysis_dir = campaign_dir / "fwhm_analysis"
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    with (analysis_dir / "fwhm_experiment_vs_simulation.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        comparison_rows = list(csv.DictReader(stream))

    runs = [run for run in manifest["runs"] if run.get("status") == "ok"]
    sweeps = []
    for index, run in enumerate(runs, start=1):
        length = float(run["pulse_length_us"])
        label = f"{length:g}".replace(".", "p")
        simulation_path = analysis_dir / f"simulation_sweep_{index:02d}_{label}us.png"
        sweeps.append(
            {
                "length": length,
                "cutoff": f"{float(manifest['cutoff']):g}",
                "experimentSrc": image_data_uri(Path(run["figure"])),
                "simulationSrc": (
                    image_data_uri(simulation_path) if simulation_path.is_file() else None
                ),
            }
        )
    t2_star_s = float(summary["t2_star_s"])
    t2_limit_hz = float(summary["t2_limit_hz"])
    rms_path = analysis_dir / "fwhm_rms_vs_pulse_length_good_snr.png"
    rms_analysis_section = ""
    if rms_path.is_file():
        rms_analysis_section = (
            "<h3 id='length-fwhm'>FWHM versus pulse length</h3>"
            "<p class='lead'>RMS FWHM uses one contiguous, scan-resolved "
            "good-SNR amplitude band per length. The plot compares experiment, "
            "matched shaped-pulse simulation, and the T2*-limited constant-pulse "
            "reference; the dashed horizontal line at 1 is the T2* floor.</p>"
            "<div>"
            f"<div class='card figure-card'><img src='{image_data_uri(rms_path)}' "
            "alt='FWHM RMS versus pulse length'><div class='caption'>RMS FWHM "
            "and experiment–simulation error versus pulse length, with the "
            "weak-drive constant-pulse simulation overlaid.</div></div>"
            "</div>"
        )
    html = REPORT_TEMPLATE.substitute(
        qubit=manifest["qubit"],
        cutoff=f"{float(manifest['cutoff']):g}",
        length_count=len(runs),
        frequency_points=int(manifest["frequency_points"]),
        amplitude_points=int(manifest["amplitude_points"]),
        frequency_span_mhz=f"{float(manifest['frequency_span_mhz']):g}",
        created_at=manifest["created_at"].replace("T", " "),
        t2_star_us=f"{t2_star_s * 1e6:.3f}",
        t2_limit_khz=f"{t2_limit_hz / 1e3:.3f}",
        finite_comparisons=summary["finite_comparisons"],
        comparison_points=summary["comparison_points"],
        overlay_image=image_data_uri(
            analysis_dir / "fwhm_experiment_vs_simulation_by_length.png"
        ),
        parity_image=image_data_uri(
            analysis_dir / "fwhm_measured_vs_simulated_parity.png"
        ),
        rms_analysis_section=rms_analysis_section,
        length_rows=build_length_rows(runs, comparison_rows),
        sweeps_json=json.dumps(sweeps, separators=(",", ":")),
    )
    output = (args.output or campaign_dir / "pulse_length_report.html").resolve()
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
