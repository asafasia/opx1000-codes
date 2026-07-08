# Shaped Pulse Spectroscopy

This project is a frequency-versus-amplitude 2D sweep that mirrors the power
Rabi chevron, but the qubit drive operation is a user-length Lorentzian-like
waveform instead of a square, cosine, or DRAG pulse.

Reusable waveform, analysis, and plotting helpers live in
`shaped_pulse_spectroscopy/`. Runnable lab entrypoints live in `experiments/`.
Clean command wrappers live in `scripts/`. The PC-only QuTiP simulator lives in
`simulation/qutip/`.

The standard Lorentzian pulse envelope is

```text
A / (1 + (t / tau)^2)
```

The root-Lorentzian pulse envelope is

```text
A / sqrt(1 + (t / tau)^2)
```

where `tau` is derived from the requested edge cutoff:

```text
t_cut = lorentzian_length_in_ns / 2
tau = t_cut / sqrt(1 / cutoff^2 - 1)
```

The Gaussian pulse envelope is

```text
A * exp(-0.5 * (t / sigma)^2)
```

where `sigma` is derived from the requested edge cutoff:

```text
t_cut = lorentzian_length_in_ns / 2
sigma = t_cut / sqrt(2 * log(1 / cutoff))
```

For all pulse shapes, `t` is centered on the midpoint of the pulse. The key
parameters are:

- `pulse_shape`: `lorentzian`, `root_lorentzian`, or `gaussian`.
- `lorentzian_length_in_ns`: total waveform length.
- `waveform_template_length_in_ns`: optional shorter stored waveform length;
  QUA stretches this template to `lorentzian_length_in_ns` with `duration`.
- `lorentzian_tau_in_ns`: standard Lorentzian width parameter.
- `cutoff`: shared edge/peak amplitude ratio for root-Lorentzian and Gaussian pulses.
- `echo`: when `True`, multiply the waveform by a midpoint sign flip so the
  first half is positive and the second half is negative.
- `lorentzian_peak_amplitude`: unscaled peak amplitude `A`.
- `min_amp_factor`, `max_amp_factor`, `amp_factor_step`: y-axis amplitude sweep.
- `frequency_span_in_mhz`, `frequency_step_in_mhz`: x-axis detuning sweep.

The plot shows detuning on the lower x-axis and absolute RF frequency on the
upper x-axis. Its left y-axis shows the equivalent Rabi frequency in MHz,
calibrated from the square `x180` pi pulse:

```text
pi_amp_hz = 1 / (2 * t_pi)
general_amp_hz = (general_amp / pi_amp) * pi_amp_hz
```

The right y-axis shows the absolute Lorentzian peak amplitude in V. When a
qubit has a T2 value, dashed vertical lines mark `+-1 / (2 * pi * T2)`.
Figures include a compact parameter banner with the pulse shape, pulse length,
cutoff or tau, echo flag, peak amplitude, sweep span/step, and square pi pulse.
When available, the banner also includes T1, T2, and `1 / (pi * T2)` in Hz.
For each amplitude in the 2D spectroscopy scan, analysis fits a Gaussian versus
detuning, stores the fitted FWHM and signal amplitude, and overlays the fitted
FWHM edges as paired markers on the heatmap.

Run the sweep with:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_2d_sweep.py
```

For fixed-amplitude detuning spectroscopy, run one selected Rabi amplitude with:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_fixed_amplitude.py
```

or loop over the plotted amplitudes for both no-echo and echo with:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_fixed_amplitude_set.py
```

The set runner defaults to `cutoff=0.005`, `20 us` pulse/template length,
`100` shots, `100` detuning points across `1 MHz`, and Rabi amplitudes
`2.32`, `4.64`, `7.58`, and `11.45 MHz`. Each amplitude is translated through
the selected qubit's square `x180` pulse into the Lorentzian amplitude
prefactor used by QUA.

The minimalist amplitude-only version keeps detuning at zero and sweeps only
the Lorentzian amplitude:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_amplitude_sweep.py
```

To sweep the root-Lorentzian cutoff itself, run:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_cutoff_sweep.py
```

This runs ten log-spaced cutoff values from `1e-4` to `0.99`. For each cutoff it
runs the class-based echo-Lorentzian spectroscopy experiment in memory, without
saving the individual per-cutoff experiments. The wrapper writes only the
combined `cutoff_sweep_fit_results.csv`, `cutoff_sweep_best_signal.csv`,
`manifest.json`, `cutoff_sweep_summary.png`, `cutoff_sweep_fwhm_heatmap.png`,
and `cutoff_sweep_per_cutoff_traces.png` under
`data/echo_lorentzian_cutoff_sweep/`. Each completed cutoff also saves only its
individual figure PNGs under `individual_figures/`; the full inner experiment
data is not saved. If the sweep is interrupted, completed rows, aggregate
figures, individual figures, and `manifest.json` are still written, with
`interrupted=true` in the manifest. The per-cutoff trace figure is the
sanity-check view: it plots the fitted FWHM and fitted signal versus equivalent
Rabi frequency for every cutoff. The heatmap shows FWHM versus equivalent Rabi
frequency in MHz and cutoff, plus a second subplot of FWHM divided by fitted
signal amplitude, with cutoff plotted on a log scale. The Rabi-frequency axis is
translated from the Lorentzian peak amplitude using the qubit's square `x180`
calibration. FWHM values in the
summary and heatmap are normalized by the qubit's T2 FWHM limit `1 / (pi*T2)`;
for example, `T2=6.86 us` corresponds to a `46.4 kHz` normalization unit.

For long pulses, keep `lorentzian_length_in_ns` as the physical pulse duration
and set `waveform_template_length_in_ns` to a shorter template, for example
`2000` ns. The experiment will store the shorter arbitrary waveform and play it
with QUA `duration=lorentzian_length_in_ns // 4`.

## Robust Cutoff Optimization Roadmap

`cutoff_optimization.py` is a key experiment. It scans the root-Lorentzian
cutoff value and, for each cutoff, runs the full 2D detuning-versus-amplitude
sweep. For every cutoff and amplitude, the analysis extracts a spectroscopy
trace versus detuning and estimates its FWHM.

The current FWHM extraction is based on a single Gaussian fit for each
amplitude. This is useful as a first pass, but it is not robust enough for all
observed traces. Real traces may contain:

- a narrow peak;
- a narrow dip;
- a broad peak;
- a broad dip;
- a feature partly outside the scanned detuning window;
- a feature narrower than the detuning step;
- a flat or noisy trace;
- distorted or multi-feature structure.

The robust analysis should not force all of these cases into one Gaussian
number. It should report both the best estimate and whether that estimate is
trustworthy.

### Target Analysis

The next analysis layer should keep the existing Gaussian fit for continuity,
but add a primary robust FWHM estimate:

1. Smooth each detuning trace lightly.
2. Estimate the baseline from the trace edges or robust percentiles.
3. Detect whether the dominant feature is a peak or a dip.
4. Find the strongest excursion from baseline.
5. Compute half-height crossings directly from the measured data.
6. Interpolate the left and right crossings.
7. Return FWHM, center, signal, polarity, and quality flags.

After the direct FWHM estimate, optional bounded model fits can be compared:

- Gaussian peak or dip;
- Lorentzian peak or dip;
- direct half-height FWHM fallback.

The selected result should be the most physically plausible and best-scored
candidate, not simply the fit with the largest signal.

### Quality Flags

Each amplitude point should carry explicit status information. Suggested flags:

- `ok`: a reliable feature was found.
- `low_signal`: the feature is too small compared with noise.
- `edge_clipped`: the feature reaches the scan boundary.
- `too_broad_for_scan`: the scan window is too narrow to measure FWHM.
- `too_narrow_for_resolution`: the detuning step is too coarse.
- `fit_failed`: the model fit did not converge or failed sanity checks.
- `multi_feature`: more than one strong feature is present.
- `flat_trace`: no meaningful spectroscopy feature is visible.

These flags are as important as the FWHM number. A clipped or unresolved trace
should not silently become a confident linewidth.

### Dataset Fields

The robust analysis should add new fields alongside the current Gaussian fields:

```text
robust_center_hz
robust_fwhm_hz
robust_fwhm_left_hz
robust_fwhm_right_hz
robust_signal
robust_polarity
robust_quality
robust_status
best_model
```

The existing Gaussian fields should remain available for comparison:

```text
gaussian_center_hz
gaussian_fwhm_hz
gaussian_fit_amplitude
gaussian_fit_abs_amplitude
gaussian_fit_r_squared
```

### Cutoff Ranking

The cutoff sweep should rank candidate cutoff values by a reliability-aware
score. A good cutoff is not only the point with the largest fitted signal. It
should have:

- small FWHM in T2-limit units;
- enough signal;
- high quality status;
- stable behavior over nearby amplitudes;
- no edge clipping;
- no resolution warning.

A future score can combine these terms:

```text
score = narrow_linewidth_reward
        + signal_reward
        + fit_quality_reward
        - edge_clipped_penalty
        - low_confidence_penalty
        - instability_penalty
```

The output should clearly distinguish the best measured point from points that
need a follow-up scan.

### Recommended Outputs

The cutoff optimization should write human-readable tables:

```text
cutoff_sweep_fit_results.csv
cutoff_sweep_best_points.csv
cutoff_sweep_quality_summary.csv
cutoff_sweep_rescan_recommendations.csv
manifest.json
```

Recommended figures:

```text
fwhm_heatmap.png
signal_heatmap.png
quality_heatmap.png
polarity_heatmap.png
best_cutoff_summary.png
per_cutoff_fwhm_traces.png
per_cutoff_signal_traces.png
diagnostic_examples.png
```

The most important new figure is `diagnostic_examples.png`. It should show
representative raw traces with the smoothed trace, baseline, detected peak or
dip, half-height crossing points, selected model, FWHM, and status flag. This
makes it easy to see whether the algorithm is making a sensible decision.

### Follow-Up Scan Recommendations

When the data does not support a trustworthy FWHM, the analysis should say what
to do next:

- If `edge_clipped` or `too_broad_for_scan`, rerun with a wider detuning span.
- If `too_narrow_for_resolution`, rerun with a smaller detuning step.
- If `low_signal`, increase averaging or avoid that amplitude/cutoff region.
- If `multi_feature`, inspect the raw trace and avoid reducing it to one FWHM.

This keeps the experiment useful even when the first scan cannot produce a
single reliable linewidth.

### Cutoff Regions And Scan Domains

Cutoff values should be treated as different experimental regimes. A single
detuning span is not appropriate for every cutoff, because high cutoffs behave
closer to square pulses and can produce broad features, while very small
cutoffs can produce much narrower features.

Recommended cutoff regions:

```text
high_cutoff:   0.1    <= cutoff <= 1
medium_cutoff: 0.001  <= cutoff <  0.1
small_cutoff:  0.0005 <= cutoff <  0.001
```

The cutoff optimization should save results into categorized folders so the
data remains easy to inspect and compare:

```text
data/echo_lorentzian_cutoff_sweep/
  high_cutoff/
    20260708_153000/
      domain_100mhz/
      domain_10mhz/
      domain_1mhz/
      region_summary.csv
      region_summary.png
      manifest.json
  medium_cutoff/
    20260708_164500/
      domain_100mhz/
      domain_10mhz/
      domain_1mhz/
      domain_0p1mhz/
      region_summary.csv
      region_summary.png
      manifest.json
  small_cutoff/
    20260708_181000/
      domain_10mhz/
      domain_1mhz/
      domain_0p1mhz/
      domain_0p01mhz/
      region_summary.csv
      region_summary.png
      manifest.json
```

Each `domain_*` folder should contain the outputs for one detuning-span choice:

```text
domain_10mhz/
  cutoff_sweep_fit_results.csv
  cutoff_sweep_best_points.csv
  cutoff_sweep_quality_summary.csv
  cutoff_sweep_rescan_recommendations.csv
  cutoff_sweep_fwhm_heatmap.png
  cutoff_sweep_signal_heatmap.png
  cutoff_sweep_quality_heatmap.png
  cutoff_sweep_per_cutoff_traces.png
  individual_figures/
  manifest.json
```

The individual 2D figures saved under `individual_figures/` should always show
the FWHM result from the fitter directly on the 2D heatmap. This makes every
domain scan self-contained: opening the figure should show the raw 2D data,
the detected linewidth markers, and enough metadata to understand the cutoff,
detuning span, detuning step, pulse length, and amplitude axis.

Suggested domain scans:

```text
high_cutoff:
  purpose: effective square-pulse-like behavior; expect broad linewidths
  cutoff range: 0.1 to 1
  detuning spans: 100 MHz, 10 MHz, 1 MHz
  workflow: start broad, then use narrower domains only when the feature is not clipped

medium_cutoff:
  purpose: mixed regime; both broad and narrow linewidths are possible
  cutoff range: 0.001 to 0.1
  detuning spans: 100 MHz, 10 MHz, 1 MHz, 0.1 MHz
  workflow: compare domains and let quality flags decide which domain is reliable

small_cutoff:
  purpose: very small cutoff; expect potentially narrow features
  cutoff range: 0.0005 to 0.001
  detuning spans: 10 MHz, 1 MHz, 0.1 MHz, 0.01 MHz
  workflow: prioritize fine resolution, but keep a wider guard scan to detect clipping
```

In the final version, `cutoff_optimization.py` should support a region/domain
configuration such as:

```text
region = high_cutoff
cutoffs = logspace(0.1, 1)
domains = [
  {span_mhz: 100, step_mhz: 0.2},
  {span_mhz: 10,  step_mhz: 0.02},
  {span_mhz: 1,   step_mhz: 0.002},
]
```

The analysis should then decide which domain produced the most trustworthy FWHM
for each cutoff and amplitude. Broad scans protect against clipped features;
fine scans protect against unresolved narrow features. The final summary should
record both the selected FWHM and the domain that produced it.

Cutoff-region campaign scripts live under:

```text
Projects/shaped_pulse_spectroscopy/scripts/cutoff_scans/
```

Preset region commands:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_high_cutoff.py
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_medium_cutoff.py
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_small_cutoff.py
```

By default these commands only print the planned scan. Add `--execute` to run
the hardware or simulation campaign:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_high_cutoff.py --qubit q1 --execute
```

Generic region/domain commands:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_region.py --region high_cutoff --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_domain.py --region high_cutoff --domain domain_10mhz --qubit q1
```

Existing campaign folders can be summarized or replotted without rerunning:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\build_region_summary.py data\echo_lorentzian_cutoff_sweep\high_cutoff\20260708_153000
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\replot_region.py data\echo_lorentzian_cutoff_sweep\high_cutoff\20260708_153000
```

The region/domain definitions live in:

```text
Projects/shaped_pulse_spectroscopy/configs/cutoff_regions.json
```
