"""Build a self-contained HTML report for a DRAG-beta calibration run."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drag_beta_kappa_calibration import evaluate_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <run_dir>/drag_beta_results.html)",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=None,
        help="Directory containing drag_kappa_joint_01 ... drag_kappa_joint_08",
    )
    parser.add_argument(
        "--extra-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional compatible run directory to append to the selector",
    )
    return parser.parse_args()


def finite(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def moving_average(values: list[float], width: int = 7) -> list[float]:
    radius = width // 2
    padded = [values[0]] * radius + values + [values[-1]] * radius
    return [sum(padded[index : index + width]) / width for index in range(len(values))]


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    points = []
    for rabi, center_hz, accepted in zip(
        record["rabi_frequency_mhz"],
        record["center_hz_vs_amplitude"],
        record["center_fit_accepted"],
        strict=True,
    ):
        rabi_value = float(rabi)
        center_khz = float(center_hz) / 1e3
        if rabi_value >= 5.0 and bool(accepted) and math.isfinite(center_khz):
            points.append([rabi_value, center_khz])

    centers = finite([point[1] for point in points])
    mean_center = sum(centers) / len(centers)
    centered = [value - mean_center for value in centers]
    smooth = moving_average(centers)
    wiggle = [value - smooth_value for value, smooth_value in zip(centers, smooth, strict=True)]
    return {
        "beta": float(record["drag_beta"]),
        "points": points,
        "accepted_points": len(points),
        "absolute_rms_khz": rms(centers),
        "centered_rms_khz": rms(centered),
        "local_wiggle_rms_khz": rms(wiggle),
        "mean_center_khz": mean_center,
        "contrast": float(record["spectroscopy_contrast"]),
        "leakage_available": bool(record["leakage_available"]),
    }


def load_raw_sweeps(
    calibration_root: Path, count: int, *, run_start: datetime
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, Path]]:
    sweeps = []
    for index in range(1, count + 1):
        experiment_root = calibration_root / f"drag_kappa_joint_{index:02d}"
        candidates = sorted(
            path
            for path in experiment_root.iterdir()
            if path.is_dir()
            and (path / "sweep.npz").is_file()
            and (path / "results.npz").is_file()
        )
        dated_candidates = []
        for path in candidates:
            try:
                timestamp = datetime.strptime(
                    f"{calibration_root.name}_{path.name}", "%Y-%m-%d_%H-%M-%S-%f"
                )
            except ValueError:
                continue
            if timestamp >= run_start:
                dated_candidates.append((timestamp, path))
        if not dated_candidates:
            raise FileNotFoundError(f"No saved sweep found below {experiment_root}")
        run_path = min(dated_candidates, key=lambda item: item[0])[1]
        with np.load(run_path / "sweep.npz") as axes, np.load(
            run_path / "results.npz"
        ) as results:
            detuning_mhz = np.asarray(axes["detuning"], dtype=float) / 1e6
            amplitude = np.asarray(axes["amp_prefactor"], dtype=float)
            state = np.asarray(results["state"], dtype=float)[0]
        sweeps.append((detuning_mhz, amplitude, state, run_path))
    return sweeps


def sweep_image_data_uri(
    *,
    beta: float,
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    state: np.ndarray,
    center_hz: list[Any],
    accepted: list[Any],
    vmin: float,
    vmax: float,
) -> str:
    figure, axis = plt.subplots(figsize=(9.4, 5.7), constrained_layout=True)
    mesh = axis.pcolormesh(
        detuning_mhz,
        rabi_mhz,
        state.T,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    centers_mhz = np.asarray(center_hz, dtype=float) / 1e6
    fit_mask = np.asarray(accepted, dtype=bool) & np.isfinite(centers_mhz)
    axis.plot(
        centers_mhz[fit_mask],
        rabi_mhz[fit_mask],
        color="white",
        linewidth=2.0,
        label="accepted fitted center",
    )
    axis.plot(
        centers_mhz[fit_mask],
        rabi_mhz[fit_mask],
        color="#17282b",
        linewidth=0.7,
    )
    axis.axvline(0, color="white", linewidth=0.8, alpha=0.55)
    axis.set(
        title=f"Measured q6 detuning × amplitude sweep · β={beta:g}",
        xlabel="Drive detuning (MHz)",
        ylabel="Rabi frequency (MHz)",
    )
    axis.legend(loc="upper left", frameon=True, fontsize=9)
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015)
    colorbar.set_label("Measured excited-state probability")
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def simulation_image_data_uri(
    *,
    beta: float,
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    probability: np.ndarray,
    vmin: float,
    vmax: float,
    title: str = "Qutrit simulation",
    colorbar_label: str = "Final P(|1>) + P(|2>)",
    cmap: str = "viridis",
) -> str:
    figure, axis = plt.subplots(figsize=(9.4, 5.7), constrained_layout=True)
    mesh = axis.pcolormesh(
        detuning_mhz,
        rabi_mhz,
        probability,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.axvline(0, color="white", linewidth=0.8, alpha=0.55)
    axis.set(
        title=f"{title} · β={beta:g}",
        xlabel="Drive detuning (MHz)",
        ylabel="Rabi frequency (MHz)",
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015)
    colorbar.set_label(colorbar_label)
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pending_simulation_image_data_uri(beta: float, label: str) -> str:
    figure, axis = plt.subplots(figsize=(9.4, 5.7), constrained_layout=True)
    axis.set_facecolor("#f4f6f7")
    axis.text(
        0.5,
        0.53,
        "Simulation pending",
        ha="center",
        va="center",
        fontsize=18,
        color="#53616a",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.45,
        f"{label} · β={beta:g}",
        ha="center",
        va="center",
        fontsize=11,
        color="#7a878e",
        transform=axis.transAxes,
    )
    axis.set_axis_off()
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=100)
    plt.close(figure)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fit_simulation_centers(
    detuning_mhz: np.ndarray,
    rabi_mhz: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    dataset = xr.Dataset(
        {
            "state": (
                ("qubit", "detuning", "amp_prefactor"),
                probability.T[np.newaxis, ...],
            )
        },
        coords={
            "qubit": ["simulation"],
            "detuning": np.asarray(detuning_mhz, dtype=float) * 1e6,
            "amp_prefactor": np.arange(rabi_mhz.size, dtype=float),
        },
    )
    metric = evaluate_pair(
        dataset,
        qubit="simulation",
        rabi_frequency_mhz=np.asarray(rabi_mhz, dtype=float),
    )
    return (
        np.asarray(metric["center_hz_vs_amplitude"], dtype=float),
        np.asarray(metric["center_fit_accepted"], dtype=bool),
        float(metric["spectroscopy_contrast"]),
    )


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>q6 DRAG-beta calibration</title>
<style>
:root{--ink:#172026;--muted:#53616a;--line:#c7d0d5;--soft:#f4f6f7;--blue:#1769aa;--orange:#c4511a;--teal:#087f8c;--gold:#8d6508}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:var(--ink);font:14px/1.55 Arial,Helvetica,sans-serif}
header{max-width:1420px;margin:auto;padding:34px 28px 22px;border-bottom:2px solid var(--ink)}.kicker{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}h1{font:600 clamp(30px,4vw,46px)/1.15 Georgia,serif;margin:5px 0 9px;letter-spacing:-.02em}header p{margin:0;color:var(--muted);max-width:980px}.runline{margin-top:14px;font-size:12px;color:var(--muted);font-family:Consolas,monospace}
main{max-width:1420px;margin:auto;padding:4px 28px 70px}section{margin:34px 0 42px}section>h2{font:600 24px/1.25 Georgia,serif;margin:0 0 7px;border-bottom:1px solid var(--line);padding-bottom:7px}.lead{color:var(--muted);margin:0 0 15px;max-width:1000px}.card{border:1px solid var(--line);background:#fff}.explorer{padding:14px}.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-right:0}.summary-item{padding:14px 16px;border-right:1px solid var(--line)}.summary-item strong{display:block;font-size:22px;font-weight:600;color:var(--blue);font-variant-numeric:tabular-nums}.summary-item span{display:block;color:var(--muted);font-size:12px;margin-top:2px}.analysis{border-left:3px solid var(--blue);padding:4px 0 4px 16px;margin-top:18px;max-width:1100px}.analysis p{margin:7px 0}.analysis b{font-weight:700}
.beta-buttons{display:flex;gap:5px;overflow:auto;padding-bottom:11px}.beta-button{border:1px solid #89969e;background:#fff;color:var(--ink);padding:6px 9px;font:12px Consolas,monospace;white-space:nowrap;cursor:pointer}.beta-button:hover{background:var(--soft)}.beta-button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:#fff}.chart-wrap{background:#fff;overflow:hidden}.chart-wrap svg{display:block;width:100%;height:auto;min-height:320px}.sweep-image{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--line)}.selected-metrics{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-right:0;margin-top:12px}.metric{padding:12px 14px;border-right:1px solid var(--line)}.metric strong{display:block;font-size:19px;font-weight:600;color:var(--blue);font-variant-numeric:tabular-nums}.metric span{color:var(--muted);font-size:11px}
.viewer-pair{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.viewer-panel{min-width:0}.viewer-panel h3{font-size:13px;margin:0 0 6px;color:var(--ink);font-weight:700}.figure-note{font-size:11px;color:var(--muted);margin:8px 0 0}
.theory-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.theory-block{border:1px solid var(--line);padding:16px 18px}.theory-block h3{font-size:15px;margin:0 0 8px}.theory-block p{margin:8px 0;color:#304047}.equation{background:var(--soft);border-left:3px solid var(--teal);padding:10px 12px;font:15px/1.6 Georgia,serif}.comparison{margin-top:14px;min-width:720px}.comparison th,.comparison td{text-align:left;vertical-align:top}.comparison th:first-child,.comparison td:first-child{width:18%;color:var(--muted)}.theory-note{margin-top:14px;padding:12px 14px;border:1px solid var(--line);background:#f7faf9}.theory-note p{margin:5px 0}
.table-wrap{overflow:auto;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;min-width:900px}th,td{padding:7px 10px;border-bottom:1px solid #dbe1e4;text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{background:var(--soft);font-size:11px;position:sticky;top:0}tbody tr:hover{background:#f6f9fb}tr.best{background:#e8f2f8}tr.balanced{box-shadow:inset 3px 0 var(--orange)}
.methods{display:grid;grid-template-columns:1fr 1fr;gap:26px}.methods h3{font-size:14px;margin:0 0 8px}.methods dl{display:grid;grid-template-columns:1fr auto;gap:5px 14px;margin:0}.methods dt{color:var(--muted)}.methods dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.limitations{margin-top:16px;padding:12px 14px;background:var(--soft);border:1px solid var(--line)}code{font-family:Consolas,monospace;font-size:.95em}.footer{color:var(--muted);font-size:11px;border-top:1px solid var(--line);padding-top:12px;margin-top:30px;word-break:break-all}
@media(max-width:900px){.summary-grid{grid-template-columns:1fr 1fr}.viewer-pair,.methods,.theory-grid{grid-template-columns:1fr}.selected-metrics{grid-template-columns:1fr 1fr}.chart-wrap svg{min-height:270px}}
@media(max-width:520px){header,main{padding-left:14px;padding-right:14px}.summary-grid,.selected-metrics{grid-template-columns:1fr}.summary-grid,.selected-metrics{border-right:1px solid var(--line)}.summary-item,.metric{border-right:0;border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<header>
  <div class="kicker">Experimental calibration report</div>
  <h1>q6 DRAG coefficient calibration</h1>
  <p>Detuning–amplitude spectroscopy of a 10 µs root-Lorentzian echo pulse, compared with a dissipative three-level transmon simulation.</p>
  <div class="runline">Run __RUN_DATE__ · κ = 0 · cutoff = __CUTOFF__ · __N_BETAS__ β values · __DETUNING_POINTS__ × __AMPLITUDE_POINTS__ grid · __NUM_SHOTS__ averages · simulation __SIM_COUNT__/__N_BETAS__</div>
</header>
<main>
  <section>
    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="summary-item"><strong>β = __BEST_CENTER_BETA__</strong><span>minimum measured centered RMS · __BEST_CENTER_RMS__ kHz</span></div>
      <div class="summary-item"><strong>β = __BEST_WIGGLE_BETA__</strong><span>minimum local-wiggle RMS · __BEST_WIGGLE_RMS__ kHz</span></div>
      <div class="summary-item"><strong>β = __BALANCED_BETA__</strong><span>balanced candidate · contrast __BALANCED_CONTRAST__</span></div>
      <div class="summary-item"><strong>β = __BEST_SIM_BETA__</strong><span>simulation RMS minimum · __BEST_SIM_RMS__ kHz</span></div>
    </div>
    <div class="analysis">
      <p><b>Measured dependence.</b> The low-oscillation region spans approximately β = __BASIN_LOW__ to __BASIN_HIGH__. The centered-RMS and local-wiggle minima occur at different sampled β values, so the result is a shallow operating region rather than a unique optimum.</p>
      <p><b>Working-point trade-off.</b> β = __BALANCED_BETA__ has the highest contrast among points lying within 15% of both measured RMS minima. It is therefore the strongest spectroscopy candidate in the low-RMS basin, but it is not yet a leakage-qualified optimum.</p>
      <p><b>Model comparison.</b> The simulation uses <code>β<sub>sim</sub> = −β<sub>hardware</sub></code>. __SIM_SCOPE__ RMS minimum is at β = __BEST_SIM_BETA__, displaced by __MODEL_SHIFT__ from the measured centered-RMS minimum. The mismatch indicates that the three-level model captures the sign trend but not the observed optimum quantitatively.</p>
    </div>
  </section>

  <section>
    <h2>1. DRAG and κ corrections</h2>
    <p class="lead">DRAG and κ both compensate drive-induced errors, but they modify different parts of the control field and should not be interpreted as the same correction.</p>
    <div class="theory-grid">
      <div class="theory-block">
        <h3>DRAG: derivative quadrature</h3>
        <div class="equation">Ω(t) = Ω<sub>I</sub>(t) + iΩ<sub>Q</sub>(t)<br>Ω<sub>Q</sub>(t) = −β [dΩ<sub>I</sub>(t)/dt] / [2π|α|]</div>
        <p>The hardware convention is <b>I + iQ</b>. Here <i>t</i> is physical time in µs, Ω<sub>I</sub> and |α| are in cyclic MHz, and β is dimensionless. β sets the size and sign of the orthogonal drive; β = 0 exactly recovers the original I-only pulse.</p>
        <p>The derivative is taken after constructing the complete signed echo waveform. It therefore includes the smooth 16 ns midpoint inversion as well as the outer pulse edges, without differentiating an instantaneous sign jump.</p>
      </div>
      <div class="theory-block">
        <h3>κ: amplitude-squared detuning</h3>
        <div class="equation">Δf<sub>κ</sub>(t) = κ Ω<sub>I</sub><sup>2</sup>(t)<br>φ<sub>κ</sub>(t) = 2π ∫<sub>0</sub><sup>t</sup> Δf<sub>κ</sub>(t′) dt′</div>
        <p>κ changes the instantaneous drive frequency to compensate the AC-Stark shift. Because the correction depends on Ω<sub>I</sub><sup>2</sup>, it has the <b>same sign on both echo halves</b> even though Ω<sub>I</sub> changes sign at the midpoint. For fixed-sign κ, its phase accumulates continuously through the inversion.</p>
        <p>This implementation supplies Δf<sub>κ</sub>(t) as a piecewise-linear QUA chirp. It is not also baked into the I/Q samples, so the same phase correction is not applied twice.</p>
      </div>
    </div>
    <div class="table-wrap"><table class="comparison">
      <thead><tr><th>Property</th><th>DRAG β</th><th>AC-Stark κ</th></tr></thead>
      <tbody>
        <tr><td>Depends on</td><td>Slope dΩ<sub>I</sub>/dt</td><td>Power Ω<sub>I</sub><sup>2</sup></td></tr>
        <tr><td>Acts through</td><td>Orthogonal Q waveform</td><td>Instantaneous frequency chirp and accumulated carrier phase</td></tr>
        <tr><td>Largest effect</td><td>Pulse edges and smooth echo inversion</td><td>High-amplitude portions of the pulse</td></tr>
        <tr><td>Echo inversion</td><td>Tracks the derivative of the signed I waveform</td><td>Does not change sign, because I<sup>2</sup> is unchanged</td></tr>
      </tbody>
    </table></div>
    <div class="theory-note">
      <p><b>Equivalent phase picture.</b> If κ were represented directly in waveform samples, the <b>I + iQ</b> envelope would be (I + iQ<sub>DRAG</sub>)e<sup>−iφκ</sup>. The hardware experiment uses the equivalent real-time chirp route instead.</p>
      <p><b>This dataset.</b> κ = 0 throughout, so the measured β dependence isolates the DRAG quadrature. The simulation comparison uses the empirical display mapping β<sub>sim</sub> = −β<sub>hardware</sub>; that mapping does not alter the hardware waveform definition above.</p>
    </div>
  </section>

  <section>
    <h2>2. Global β dependence</h2>
    <p class="lead">RMS metrics use accepted fitted centers at Rabi frequency ≥ 5 MHz. Lines connect sampled β values only and do not imply an interpolation model.</p>
    <div class="card explorer">
      <div class="chart-wrap"><svg id="rmsChart" viewBox="0 0 900 430" role="img" aria-label="Resonance-center RMS versus DRAG beta"></svg></div>
    </div>
  </section>

  <section>
    <h2>3. Fitted center versus drive amplitude</h2>
    <p class="lead">Select β to compare the measured resonance-center trajectory with the corresponding sign-mapped qutrit simulation.</p>
    <div class="card explorer">
      <div class="beta-buttons" id="centerButtons"></div>
      <div class="chart-wrap"><svg id="centerChart" viewBox="0 0 900 500" role="img" aria-label="Fitted resonance center versus Rabi frequency"></svg></div>
      <div class="selected-metrics">
        <div class="metric"><strong id="metricCentered">—</strong><span>measured centered RMS</span></div>
        <div class="metric"><strong id="metricSimulationCentered">—</strong><span>simulated centered RMS</span></div>
        <div class="metric"><strong id="metricWiggle">—</strong><span>measured local wiggle RMS</span></div>
        <div class="metric"><strong id="metricContrast">—</strong><span>spectroscopy contrast</span></div>
      </div>
    </div>
  </section>

  <section>
    <h2>4. Detuning–amplitude maps</h2>
    <p class="lead">Measured excitation, corresponding simulated total excitation, and simulated |2⟩ leakage are shown on the same detuning and Rabi axes.</p>
    <div class="card explorer">
      <div class="beta-buttons" id="sweepButtons"></div>
      <div class="viewer-pair">
        <div class="viewer-panel"><h3>Measured excited-state probability</h3><img id="sweepImage" class="sweep-image" alt="Measured detuning by amplitude sweep"></div>
        <div class="viewer-panel"><h3>Simulation: P(|1⟩) + P(|2⟩)</h3><img id="simulationImage" class="sweep-image" alt="Simulated detuning by amplitude sweep"></div>
        <div class="viewer-panel"><h3>Simulation: P(|2⟩)</h3><img id="leakageImage" class="sweep-image" alt="Simulated leakage by detuning and amplitude"></div>
      </div>
      <p class="figure-note">Color scales are shared within each data type across β. Simulated leakage is a model output; three-state leakage was not measured in this acquisition.</p>
    </div>
  </section>

  <section>
    <h2>5. Quantitative results</h2>
    <p class="lead">All acquired β values. The blue row minimizes measured centered RMS; the orange marker identifies the balanced low-RMS/high-contrast candidate.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>β</th><th>Centered RMS (kHz)</th><th>Local wiggle RMS (kHz)</th><th>Mean center (kHz)</th><th>Measured contrast</th><th>Simulation RMS (kHz)</th><th>Sim. max P₂ (%)</th></tr></thead>
      <tbody>__RESULT_ROWS__</tbody>
    </table></div>
  </section>

  <section>
    <h2>6. Methods and limitations</h2>
    <div class="methods">
      <div><h3>Experiment</h3><dl>
        <dt>Qubit</dt><dd>q6</dd><dt>Pulse</dt><dd>root-Lorentzian echo</dd><dt>Duration</dt><dd>10 µs</dd><dt>Cutoff</dt><dd>__CUTOFF__</dd><dt>Midpoint transition</dt><dd>16 ns</dd><dt>Complex envelope</dt><dd>I + iQ</dd><dt>DRAG definition</dt><dd>Q = −β(dI/dt)/(2π|α|)</dd><dt>Derivative coordinate</dt><dd>physical time in µs</dd><dt>κ implementation</dt><dd>QUA real-time chirp; κ = 0</dd><dt>β range</dt><dd>__BETA_MIN__ to __BETA_MAX__</dd><dt>Grid</dt><dd>__DETUNING_POINTS__ detuning × __AMPLITUDE_POINTS__ amplitude</dd><dt>Averages</dt><dd>__NUM_SHOTS__</dd>
      </dl></div>
      <div><h3>Analysis and simulation</h3><dl>
        <dt>Center extraction</dt><dd>negative-Gaussian fit</dd><dt>Acceptance region</dt><dd>Rabi ≥ 5 MHz</dd><dt>Centered RMS</dt><dd>RMS after mean removal</dd><dt>Local wiggle</dt><dd>RMS after local trend removal</dd><dt>Model</dt><dd>dissipative 3-level transmon</dd><dt>Integrator</dt><dd>vectorized RK4</dd><dt>T₁ / T₂*</dt><dd>48.82 / 22.79 µs</dd><dt>Anharmonicity</dt><dd>−237.95 MHz</dd><dt>Simulation mapping</dt><dd>βsim = −βhardware</dd>
      </dl></div>
    </div>
    <div class="limitations"><b>Limitations.</b> This is one β sweep without independent repeats or propagated fit covariance, so no statistical confidence intervals are claimed. Three-state discrimination was unavailable; measured P<sub>f</sub> is therefore absent, and the final β should not be persisted solely from simulated leakage. A focused repeat across the low-RMS basin with direct leakage measurement is the appropriate confirmation.</div>
  </section>

  <div class="footer">Plan SHA256: __PLAN_HASH__ · Source: __RUN_DIR__</div>

</main>
<script>
const series=__SERIES__;
const best=__BEST__, control=__CONTROL__;
let selected=best;
const ns='http://www.w3.org/2000/svg';
const fmt=value=>Number.isFinite(value)?Number(value).toFixed(2)+' kHz':'pending';
const betaLabel=value=>value===0?'0':(Number.isInteger(value*10)?value.toFixed(1):value.toFixed(2));
function element(name,attrs={},text=''){const node=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));node.textContent=text;return node}
function drawRms(){
 const svg=document.getElementById('rmsChart');svg.replaceChildren();const ordered=[...series].sort((a,b)=>a.beta-b.beta);
 const W=900,H=430,m={l:70,r:22,t:30,b:58},xmin=Math.min(...ordered.map(s=>s.beta))-.05,xmax=Math.max(...ordered.map(s=>s.beta))+.05;
 const values=ordered.flatMap(s=>[s.centered_rms_khz,s.local_wiggle_rms_khz,s.simulation_centered_rms_khz]).filter(Number.isFinite);const ymax=Math.max(2,Math.ceil(Math.max(...values)/2)*2);
 const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>m.t+(ymax-y)/ymax*(H-m.t-m.b);
 for(let y=0;y<=ymax;y+=2){svg.append(element('line',{x1:m.l,x2:W-m.r,y1:Y(y),y2:Y(y),stroke:y===0?'#627477':'#d6e0dd','stroke-width':y===0?1.4:1}));svg.append(element('text',{x:m.l-12,y:Y(y)+5,'text-anchor':'end',fill:'#3e5558','font-size':13},y))}
 ordered.forEach((s,index)=>{if(index%5===0||index===ordered.length-1){svg.append(element('line',{x1:X(s.beta),x2:X(s.beta),y1:m.t,y2:H-m.b,stroke:'#e2e9e7','stroke-width':1}));svg.append(element('text',{x:X(s.beta),y:H-m.b+25,'text-anchor':'middle',fill:'#3e5558','font-size':12},betaLabel(s.beta)))}});const selectedPoint=ordered.find(s=>s.beta===selected);svg.append(element('line',{x1:X(selectedPoint.beta),x2:X(selectedPoint.beta),y1:m.t,y2:H-m.b,stroke:'#172026','stroke-width':1.3,'stroke-dasharray':'4 4',opacity:.55}));
 const traces=[
  {key:'centered_rms_khz',label:'measured centered',color:'#1769aa',dash:''},
  {key:'local_wiggle_rms_khz',label:'measured local wiggle',color:'#c4511a',dash:''},
  {key:'simulation_centered_rms_khz',label:'simulation centered',color:'#087f8c',dash:'9 6'}
 ];
 traces.forEach((trace,index)=>{const tracePoints=ordered.filter(s=>Number.isFinite(s[trace.key]));const d=tracePoints.map((s,i)=>(i?'L':'M')+X(s.beta).toFixed(1)+' '+Y(s[trace.key]).toFixed(1)).join(' ');const attrs={d,fill:'none',stroke:trace.color,'stroke-width':3,'stroke-linecap':'round','stroke-linejoin':'round'};if(trace.dash)attrs['stroke-dasharray']=trace.dash;svg.append(element('path',attrs));tracePoints.forEach(s=>svg.append(element('circle',{cx:X(s.beta),cy:Y(s[trace.key]),r:s.beta===selected?5:3.3,fill:'#fff',stroke:trace.color,'stroke-width':2.5})));const ly=42+index*24;const lineAttrs={x1:W-250,x2:W-212,y1:ly,y2:ly,stroke:trace.color,'stroke-width':3};if(trace.dash)lineAttrs['stroke-dasharray']=trace.dash;svg.append(element('line',lineAttrs));svg.append(element('text',{x:W-202,y:ly+5,fill:'#263c3e','font-size':13},trace.label))});
 svg.append(element('text',{x:W/2,y:H-10,'text-anchor':'middle',fill:'#263c3e','font-size':15},'DRAG β'));const ylab=element('text',{x:17,y:H/2,'text-anchor':'middle',fill:'#263c3e','font-size':15,transform:`rotate(-90 17 ${H/2})`},'RMS (kHz)');svg.append(ylab);
}
function draw(){
 const svg=document.getElementById('centerChart');svg.replaceChildren();
 const W=900,H=500,m={l:70,r:22,t:24,b:58};const all=series.flatMap(s=>[...s.points,...s.simulation_points]);const xmin=5,xmax=Math.max(...all.map(p=>p[0]));const ymax=Math.ceil(Math.max(24,...all.map(p=>Math.abs(p[1])))/5)*5,ymin=-ymax;
 const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>m.t+(ymax-y)/(ymax-ymin)*(H-m.t-m.b);
 for(let y=ymin;y<=ymax;y+=10){svg.append(element('line',{x1:m.l,x2:W-m.r,y1:Y(y),y2:Y(y),stroke:y===0?'#627477':'#d6e0dd','stroke-width':y===0?1.4:1}));svg.append(element('text',{x:m.l-12,y:Y(y)+5,'text-anchor':'end',fill:'#3e5558','font-size':13},y))}
 for(let x=5;x<=xmax;x+=5){svg.append(element('line',{x1:X(x),x2:X(x),y1:m.t,y2:H-m.b,stroke:'#e2e9e7'}));svg.append(element('text',{x:X(x),y:H-m.b+25,'text-anchor':'middle',fill:'#3e5558','font-size':13},x))}
 [...series].sort((a,b)=>(a.beta===selected?1:0)-(b.beta===selected?1:0)).forEach(s=>{const active=s.beta===selected,zero=s.beta===control;const d=s.points.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');svg.append(element('path',{d,fill:'none',stroke:active?'#1769aa':zero?'#172026':'#aeb9bf','stroke-width':active?3.5:zero?2.5:1.1,opacity:active||zero?1:.32,'stroke-linecap':'round','stroke-linejoin':'round'}))});
 const selectedSeries=series.find(item=>item.beta===selected);const simulationPath=selectedSeries.simulation_points.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(' ');svg.append(element('path',{d:simulationPath,fill:'none',stroke:'#087f8c','stroke-width':3,'stroke-dasharray':'9 6','stroke-linecap':'round','stroke-linejoin':'round'}));
 svg.append(element('line',{x1:W-222,x2:W-184,y1:40,y2:40,stroke:'#1769aa','stroke-width':3.5}));svg.append(element('text',{x:W-174,y:45,fill:'#263c3e','font-size':13},'measured'));
 svg.append(element('line',{x1:W-222,x2:W-184,y1:65,y2:65,stroke:'#087f8c','stroke-width':3,'stroke-dasharray':'9 6'}));svg.append(element('text',{x:W-174,y:70,fill:'#263c3e','font-size':13},'simulation'));
 svg.append(element('text',{x:W/2,y:H-10,'text-anchor':'middle',fill:'#263c3e','font-size':15},'Rabi frequency (MHz)'));const ylab=element('text',{x:17,y:H/2,'text-anchor':'middle',fill:'#263c3e','font-size':15,transform:`rotate(-90 17 ${H/2})`},'Fitted center (kHz)');svg.append(ylab);
 const s=selectedSeries;document.getElementById('metricCentered').textContent=fmt(s.centered_rms_khz);document.getElementById('metricSimulationCentered').textContent=fmt(s.simulation_centered_rms_khz);document.getElementById('metricWiggle').textContent=fmt(s.local_wiggle_rms_khz);document.getElementById('metricContrast').textContent=s.contrast.toFixed(3);document.getElementById('sweepImage').src=s.sweep_image;document.getElementById('sweepImage').alt=`Measured sweep for beta ${s.beta}`;document.getElementById('simulationImage').src=s.simulation_image;document.getElementById('simulationImage').alt=`Qutrit simulation for beta ${s.beta}`;document.getElementById('leakageImage').src=s.simulation_leakage_image;document.getElementById('leakageImage').alt=`Simulated leakage for beta ${s.beta}`;document.querySelectorAll('.beta-button').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.beta)===selected));drawRms();
}
function addButtons(container){series.forEach(s=>{const b=document.createElement('button');b.className='beta-button';b.dataset.beta=s.beta;b.textContent='β '+betaLabel(s.beta);b.onclick=()=>{selected=s.beta;draw()};container.append(b)})}
addButtons(document.getElementById('sweepButtons'));
addButtons(document.getElementById('centerButtons'));
draw();
</script>
</body></html>"""


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    plan = json.loads((run_dir / "approved_plan.json").read_text(encoding="utf-8"))
    source_dirs = [run_dir, *(path.resolve() for path in args.extra_run_dir)]
    record_groups = [
        json.loads((source / "records.json").read_text(encoding="utf-8"))
        for source in source_dirs
    ]
    records = [record for group in record_groups for record in group]
    record_sources = [
        source
        for source, group in zip(source_dirs, record_groups, strict=True)
        for _record in group
    ]
    series = [analyze_record(record) for record in records]
    repository_root = Path(__file__).resolve().parents[3]
    raw_sweeps = []
    for source, group in zip(source_dirs, record_groups, strict=True):
        calibration_root = (
            args.calibration_root.resolve()
            if args.calibration_root is not None
            else repository_root / "data" / "calibrations" / source.name[:10]
        )
        raw_sweeps.extend(
            load_raw_sweeps(
                calibration_root,
                len(group),
                run_start=datetime.strptime(source.name, "%Y-%m-%d_%H-%M-%S"),
            )
        )
    all_state = np.concatenate([sweep[2].ravel() for sweep in raw_sweeps])
    vmin, vmax = np.nanpercentile(all_state, [1.0, 99.0])
    for result, record, raw in zip(series, records, raw_sweeps, strict=True):
        detuning_mhz, _amplitude, state, source_path = raw
        result["sweep_image"] = sweep_image_data_uri(
            beta=result["beta"],
            detuning_mhz=detuning_mhz,
            rabi_mhz=np.asarray(record["rabi_frequency_mhz"], dtype=float),
            state=state,
            center_hz=record["center_hz_vs_amplitude"],
            accepted=record["center_fit_accepted"],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        result["raw_source"] = str(source_path)
    simulation_arrays = []
    for result, source in zip(series, record_sources, strict=True):
        label = (
            f"{result['beta']:+.3f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )
        simulation_path = source / "simulation" / f"beta_{label}.npz"
        if not simulation_path.is_file():
            simulation_arrays.append(None)
            continue
        with np.load(simulation_path) as simulated:
            required_base_fields = {
                "simulation_drag_beta",
                "detuning_mhz",
                "rabi_mhz",
                "total_excited",
                "second_excited",
            }
            if not required_base_fields.issubset(simulated.files) or not np.isclose(
                float(simulated["simulation_drag_beta"]),
                -float(result["beta"]),
                rtol=0.0,
                atol=1e-12,
            ):
                simulation_arrays.append(None)
                continue
            detuning_mhz = np.asarray(simulated["detuning_mhz"], dtype=float)
            rabi_mhz = np.asarray(simulated["rabi_mhz"], dtype=float)
            probability = np.asarray(simulated["total_excited"], dtype=float)
            leakage = np.asarray(simulated["second_excited"], dtype=float)
            if {
                "sim_center_hz",
                "sim_center_accepted",
                "sim_spectroscopy_contrast",
            }.issubset(simulated.files):
                sim_center_hz = np.asarray(simulated["sim_center_hz"], dtype=float)
                sim_center_accepted = np.asarray(
                    simulated["sim_center_accepted"], dtype=bool
                )
                sim_contrast = float(
                    np.asarray(simulated["sim_spectroscopy_contrast"])
                )
            else:
                sim_center_hz, sim_center_accepted, sim_contrast = (
                    fit_simulation_centers(detuning_mhz, rabi_mhz, probability)
                )
            simulation_arrays.append(
                (
                    detuning_mhz,
                    rabi_mhz,
                    probability,
                    leakage,
                    sim_center_hz,
                    sim_center_accepted,
                    sim_contrast,
                )
            )
    available_simulations = [item for item in simulation_arrays if item is not None]
    all_simulated = np.concatenate(
        [item[2].ravel() for item in available_simulations]
    )
    sim_vmin, sim_vmax = np.nanpercentile(all_simulated, [1.0, 99.0])
    leakage_vmax = float(max(item[3].max() for item in available_simulations))
    for result, simulated in zip(series, simulation_arrays, strict=True):
        if simulated is None:
            result["simulation_points"] = []
            result["simulation_centered_rms_khz"] = None
            result["simulation_contrast"] = None
            result["simulation_max_leakage_pct"] = None
            result["simulation_mean_leakage_pct"] = None
            result["simulation_image"] = pending_simulation_image_data_uri(
                result["beta"], "Total excitation"
            )
            result["simulation_leakage_image"] = pending_simulation_image_data_uri(
                result["beta"], "Leakage"
            )
            continue
        (
            detuning_mhz,
            rabi_mhz,
            probability,
            leakage,
            sim_center_hz,
            sim_accepted,
            sim_contrast,
        ) = simulated
        sim_mask = sim_accepted & np.isfinite(sim_center_hz) & (rabi_mhz >= 5.0)
        result["simulation_points"] = [
            [float(rabi), float(center / 1e3)]
            for rabi, center in zip(
                rabi_mhz[sim_mask], sim_center_hz[sim_mask], strict=True
            )
        ]
        simulated_centers_khz = sim_center_hz[sim_mask] / 1e3
        result["simulation_centered_rms_khz"] = float(
            np.sqrt(
                np.mean(
                    (simulated_centers_khz - np.mean(simulated_centers_khz)) ** 2
                )
            )
        )
        result["simulation_contrast"] = sim_contrast
        result["simulation_max_leakage_pct"] = float(np.max(leakage) * 100.0)
        result["simulation_mean_leakage_pct"] = float(np.mean(leakage) * 100.0)
        result["simulation_image"] = simulation_image_data_uri(
            beta=result["beta"],
            detuning_mhz=detuning_mhz,
            rabi_mhz=rabi_mhz,
            probability=probability,
            vmin=float(sim_vmin),
            vmax=float(sim_vmax),
        )
        result["simulation_leakage_image"] = simulation_image_data_uri(
            beta=result["beta"],
            detuning_mhz=detuning_mhz,
            rabi_mhz=rabi_mhz,
            probability=leakage,
            vmin=0.0,
            vmax=leakage_vmax,
            title="Simulated leakage",
            colorbar_label="Final P(|2>)",
            cmap="magma",
        )
    best_center = min(series, key=lambda item: item["centered_rms_khz"])
    best_wiggle = min(series, key=lambda item: item["local_wiggle_rms_khz"])
    simulated_series = [
        item
        for item in series
        if item["simulation_centered_rms_khz"] is not None
    ]
    best_simulation = min(
        simulated_series, key=lambda item: item["simulation_centered_rms_khz"]
    )
    basin = [
        item
        for item in series
        if item["centered_rms_khz"] <= 1.15 * best_center["centered_rms_khz"]
        and item["local_wiggle_rms_khz"]
        <= 1.15 * best_wiggle["local_wiggle_rms_khz"]
    ]
    if not basin:
        basin = [best_center]
    balanced = max(basin, key=lambda item: item["contrast"])
    best_beta = best_center["beta"]
    control_beta = min(series, key=lambda item: abs(item["beta"]))["beta"]
    result_rows = []
    for item in sorted(series, key=lambda entry: entry["beta"]):
        classes = []
        if item is best_center:
            classes.append("best")
        if item is balanced:
            classes.append("balanced")
        class_attribute = f' class="{" ".join(classes)}"' if classes else ""
        simulation_rms = item["simulation_centered_rms_khz"]
        simulation_leakage = item["simulation_max_leakage_pct"]
        result_rows.append(
            f"<tr{class_attribute}>"
            f"<td>{item['beta']:+.4f}</td>"
            f"<td>{item['centered_rms_khz']:.3f}</td>"
            f"<td>{item['local_wiggle_rms_khz']:.3f}</td>"
            f"<td>{item['mean_center_khz']:.3f}</td>"
            f"<td>{item['contrast']:.4f}</td>"
            f"<td>{'—' if simulation_rms is None else f'{simulation_rms:.3f}'}</td>"
            f"<td>{'—' if simulation_leakage is None else f'{simulation_leakage:.5f}'}</td>"
            "</tr>"
        )
    beta_values = [item["beta"] for item in series]
    waveform = plan["waveform"]
    grid = plan["grid"]
    replacements = {
        "__BEST__": json.dumps(best_beta),
        "__CONTROL__": json.dumps(control_beta),
        "__RUN_DATE__": run_dir.name.replace("_", " "),
        "__CUTOFF__": f"{float(waveform['cutoff']):g}",
        "__N_BETAS__": str(len(series)),
        "__SIM_COUNT__": str(len(simulated_series)),
        "__SIM_SCOPE__": (
            "Across all model points, its"
            if len(simulated_series) == len(series)
            else f"Among the {len(simulated_series)} completed model points, its provisional"
        ),
        "__DETUNING_POINTS__": str(int(grid["detuning_points"])),
        "__AMPLITUDE_POINTS__": str(int(grid["amplitude_points"])),
        "__NUM_SHOTS__": str(int(grid["num_shots"])),
        "__BEST_CENTER_BETA__": f"{best_center['beta']:+.4f}",
        "__BEST_CENTER_RMS__": f"{best_center['centered_rms_khz']:.3f}",
        "__BEST_WIGGLE_BETA__": f"{best_wiggle['beta']:+.4f}",
        "__BEST_WIGGLE_RMS__": f"{best_wiggle['local_wiggle_rms_khz']:.3f}",
        "__BALANCED_BETA__": f"{balanced['beta']:+.4f}",
        "__BALANCED_CONTRAST__": f"{balanced['contrast']:.4f}",
        "__BEST_SIM_BETA__": f"{best_simulation['beta']:+.4f}",
        "__BEST_SIM_RMS__": f"{best_simulation['simulation_centered_rms_khz']:.3f}",
        "__BASIN_LOW__": f"{min(item['beta'] for item in basin):+.4f}",
        "__BASIN_HIGH__": f"{max(item['beta'] for item in basin):+.4f}",
        "__MODEL_SHIFT__": f"{best_simulation['beta'] - best_center['beta']:+.4f} β",
        "__BETA_MIN__": f"{min(beta_values):+.4f}",
        "__BETA_MAX__": f"{max(beta_values):+.4f}",
        "__RESULT_ROWS__": "".join(result_rows),
        "__PLAN_HASH__": str(plan["plan_sha256"]),
        "__RUN_DIR__": str(run_dir),
    }
    html = (
        HTML.replace("__SERIES__", json.dumps(series, separators=(",", ":")))
    )
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    output = (args.output or run_dir / "drag_beta_results.html").resolve()
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
