# Shaped Pulse Spectroscopy

This project is a frequency-versus-amplitude 2D sweep that mirrors the power
Rabi chevron, but the qubit drive operation is a user-length Lorentzian-like
waveform instead of a square, cosine, or DRAG pulse.

Reusable waveform, analysis, and plotting helpers live in
`shaped_pulse_spectroscopy/`. Runnable lab entrypoints live in `experiments/`.
Clean command wrappers live in `scripts/`. The PC-only QuTiP simulator lives in
`simulation/qutip/`.

## Figures from the 2D echo sweep

With analysis and plotting enabled, `scripts/run_2d_sweep.py` now adds five
figures per qubit to the viridis 2D map: resonance detuning, FWHM, fitted dip
contrast versus calibrated Rabi frequency, up to eight constant-Rabi detuning
slices with fits, and constant-detuning response slices versus Rabi frequency.
Every Rabi axis uses the actual pulse peak `max(abs(I + 1j*Q))` after envelope
and echo smoothing and DRAG, multiplied by `abs(amp_prefactor)`. Conversion to
MHz uses the calibrated `x180` sampled area. The chosen nominal amplitude is
retained separately as `full_amp`; it does not set the plotted Rabi frequency.
The 2D run saves `peak_complex_amplitude_v` with the raw sweep coordinates;
offline processing of older data reconstructs it from the run parameters and
the supplied qubit calibration. Use the saved profile for historical runs.
The resonance plot
shows reference lines at ±1/(2πT₂*) and the linewidth plot at 1/(πT₂*), using
Ramsey T₂* when available. These are linewidth references, not bounds on a
physical resonance shift.

For echo/state readout, `plot_sweep_diagnostics=True` (default) runs a fast
bounded negative-Gaussian fit independently of the optional `fit_fwhm` fitter.
Contrast means fitted baseline minus dip minimum, in measured-state units.
The fit trims up to 40 samples at each frequency edge and smooths by one bin;
acceptance uses unsmoothed R² ≥ 0.1, contrast ≥ 0.02, a width of at least two
bins, and both half-height crossings inside the fit window. Rejected or missing
fits leave gaps and are marked at their Rabi positions along the lower plot edge.
Figures are registered
with the normal figure saver, and fit values/quality flags are included in
`analysis_result.json` when analysis saving is enabled. Set
`plot_sweep_diagnostics=False` to disable these additions. They are also
skipped for acquisitions with `CalibrationOptions.plot_data=False`, IQ readout,
or non-echo pulses.

## Current shared echo settings and fast beta scan

Edit `shaped_pulse_spectroscopy/echo_parameters.py` for the current echo pulse:
20,000 ns root-Lorentzian and full-length template, cutoff 0.001, nominal
amplitude 0.7 V, beta **+0.5**, 16 ns midpoint smoothing,
polynomial edge smoothing with power 5, and no AC-Stark chirp. `DEFAULT_QUBIT`
is q1. Active reset and unmitigated two-state discrimination are used.

As of 2026-10-27, echo DRAG uses the same positive-derivative beta convention
as this repository's Gaussian DRAG: `Q = +beta*dI/dt/(2*pi*abs(alpha))`.
Historical measurements used the opposite sign: replay their physical I/Q
waveforms with `beta_new = -beta_old`. Saved data are unchanged. In particular,
the historical -1.097225 optimum corresponds to +1.097225 now. Explicit beta
ranges in historical scripts/plans must be converted before replaying them.
New raw pulse metadata records `gaussian_positive_derivative_v1`.

The echo chirp and amplitude-axis conversion now use the reference pi pulse's
sampled area. This preserves calibrated voltages and corrects the former
square-pulse assumption when `x180` is cosine shaped.

To compare the four requested inversion times at beta +0.5, run
`python Projects/shaped_pulse_spectroscopy/scripts/compare_echo_inversion_times.py`
for offline checks, or add `--execute` to measure. The fixed comparison uses
4, 8, 16, and 50 ns, 25 amplitudes (0.2–1), 121 detunings (±500 kHz), and 400
shots, with no detuning chirp and no profile updates.

The general 2D, amplitude-only, fixed-amplitude, fixed-amplitude batch, kappa,
cutoff, pulse-length, and beta entrypoints now obtain their starting pulse
from this file. Generic `Parameters()` remains a neutral model; new echo
scripts should call `echo_parameters()` or `fast_echo_parameters()`.
Explicit CLI values, swept quantities, campaign configurations and saved
reproduction parameters take precedence. The named historical q6 plans keep
their explicit waveform settings; paper-reproduction and saved-data analysis
scripts continue to use their own specified/saved parameters.

`FAST_ECHO_SCAN` in the same file controls the ordinary echo 2D map and the
new beta-map preset: **13 linear amplitude factors from 0.2 to 1.0**, **81
detunings over 1 MHz**, **200 averages**. Both endpoints are included. This is
1,053 points per map versus 6,200 in the previous 31 x 200 verification scan.
Zero amplitude is excluded because it has no dip to fit. Amplitudes are
dimensionless multipliers of the shared 0.7 V nominal waveform.

The existing `scripts/optimize_echo_drag_beta.py` already acquires a 2D map at
each beta; its original three-amplitude defaults remain available. The new
`scripts/optimize_echo_drag_beta_2d.py` reuses its acquisition and fitting code
with **40 betas from 0 down to -1.5** and the shared fast grid:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta_2d.py --dry-run
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta_2d.py --execute
```

Omitting `--execute` only validates every beta's complete I/Q voltage and
generates QUA offline. Override the grid/range as needed with `--beta-min`,
`--beta-max`, `--beta-points`, `--amplitude-points`, `--frequency-points`, or
`--num-shots`. For example, `--beta-min -1.3 --beta-max -0.9 --beta-points 21`
restricts the scan near the previous optimum. Existing direct experiment
entrypoints retain their normal acquisition behavior.

At each beta the existing bounded negative-Gaussian fitter fits every
amplitude trace. With centers c_i and fit errors sigma_i, the score is:

```text
c_ref = sum(c_i / sigma_i^2) / sum(1 / sigma_i^2)
center_rms_hz = sqrt(mean((c_i - c_ref)^2))
```

This measures resonance-line variation across amplitudes after removing a
common detuning offset. It is not the residual RMS of a Gaussian fit.
Every requested amplitude must have a finite center/error, positive FWHM,
contrast >= 0.02, and R-squared >= 0.1. Otherwise that beta is rejected,
rather than benefiting from dropping a difficult amplitude. The best sampled
beta is reported only when the full sweep completes; boundary minima are
flagged. No device profile or shared default beta is updated automatically.

Results go under `data/echo_drag_beta_2d/<timestamp>/`: raw calibration folders,
`settings.json` with the resolved pulse and grids, per-beta `*_scan.npz` and
`*_map.png` (map plus fitted line), `fits.csv`, `records.json`, `summary.json`,
and `beta_summary.png`. The CSV contains center/error, FWHM, contrast,
R-squared and acceptance for every amplitude, plus the total RMS per beta.

An offline subsampling check of the 2026-10-23 q1 verification data retained
all 13 fits: 15.03 kHz RMS on the 13 x 81 grid, versus 14.96 kHz on the dense
25-amplitude fitted subset. This check used existing 1,000-average data; it
does not establish the noise at the new 200-average setting.

The standard Lorentzian pulse envelope is

```text
A / (1 + (t / tau)^2)
```

where `tau` is derived from the requested edge cutoff:

```text
t_cut = lorentzian_length_in_ns / 2
tau = t_cut / sqrt(1 / cutoff - 1)
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
- `cutoff`: shared edge/peak amplitude ratio for Lorentzian, root-Lorentzian,
  and Gaussian pulses.
- `echo`: when `True`, multiply the waveform by a midpoint sign flip so the
  first half is positive and the second half is negative.
- `drag_beta`: optional dimensionless DRAG coefficient. `0.0` disables the
  DRAG quadrature.
- `echo_transition_time_ns`: hardware-resolvable smooth echo inversion with
  or without DRAG (16 ns by default). Set to `0.0` for an abrupt inversion.
- `edge_smoothing`: `"none"` (default) or `"polynomial"` for root-Lorentzian
  pulses, with or without echo and DRAG.
- `edge_smoothing_power`: positive integer window power (default `5`).
- `stark_kappa_mhz_inv`: quadratic AC-Stark coefficient applied as a QUA
  real-time frequency chirp. It is not also applied as IQ phase modulation.
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
qubit has a Ramsey T2* value, dashed vertical lines mark
`+-1 / (2 * pi * T2*)`.

The polynomial option implements Eqs. (4)-(7) of the amplitude-robust
spectroscopy paper (`../amplitude-robust-spectrscopy/paper/main.pdf`, p. 3):

```text
f(t) = 1 / sqrt(1 + (t/tau)^2)
tau = (L/2) / sqrt(c^(-2) - 1)
g(t) = (f(t)-c)/(1-c) * [1-(2*t/L)^2]^p   for |t| < L/2
g(t) = 0                                otherwise
I(t) = A*g(t)*s(t)                       with echo enabled
```

For example:

```python
parameters.pulse_shape = "root_lorentzian"
parameters.edge_smoothing = "polynomial"
parameters.edge_smoothing_power = 5
parameters.echo = True
parameters.echo_transition_time_ns = 16.0
```

The cutoff now sets the underlying core width; the corrected endpoints are
zero. At `c=1` the continuous limit is `[1-(2*t/L)^2]^(p+1)`. The continuous
unsigned center remains one, with no area or sampled-peak normalization.
The existing envelope grid includes both endpoints; the midpoint transition
uses centered sample times with spacing `L/N`. This small sampling convention
difference is retained for compatibility. Stretched templates must still
resolve the transition with at least four samples. For even sample counts,
the signed echo remains antisymmetric and its discrete area cancels.

`WaveformPulse` defines the complex envelope as `I + iQ`. DRAG differentiates
the complete smooth signed I waveform with respect to physical time:

```text
Q = -drag_beta * dI/dt_us / (2*pi*abs(anharmonicity_MHz))
```

The paper's supplemental pulse definition uses `I - iQ` with this same Q
formula. For the same complex-envelope representation this reverses the
beta sign; do not directly transfer a paper beta value to hardware without
checking the mixer/Hamiltonian convention. Edge smoothing does not alter
the existing hardware Q convention. DRAG differentiates the complete
floor-corrected, windowed, signed envelope, including both edge and midpoint
contributions.

Both lanes are arbitrary-waveform samples in volts and QUA's amplitude
prefactor scales I and Q together. The pulse factory checks
`max(abs(I + 1j*Q))` after the largest sweep prefactor against the existing
1 V OPX waveform bound and records the peak and remaining headroom.

To optimize DRAG beta using 1D spectroscopy slices at several amplitudes, run:

```powershell
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta.py --dry-run
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta.py --execute
```

The default is an offline check. Acquisition requires `--execute`. The script
uses the existing `EchoLorentzian` calibration lifecycle and negative-Gaussian
dip fit. By default it acquires three amplitude prefactors (`0.25`, `0.5`, `1.0`)
together at each of 40 equally spaced betas from `-0.7` through `+0.1` (120
slices total). The winning sampled beta minimizes the RMS of all fitted
centers about their inverse-variance-weighted mean. A common frequency offset
is therefore allowed; amplitude-dependent resonance motion is penalized.
All amplitude fits must have finite positive center uncertainty, contrast at least
`0.02`, and R-squared at least `0.1`; a beta with any failed slice cannot win.

Defaults are q1, 20 us, cutoff `0.001`, a 16 ns midpoint transition, polynomial
edge power 5, no AC-Stark chirp, 201 frequencies over 1 MHz, and 200 shots. The
waveform's nominal scale is `0.7 V`; the amplitudes are dimensionless
prefactors, not voltages. Every complete I/Q waveform is checked against the
1 V waveform bound before connecting. Examples of overrides:

```powershell
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta.py --dry-run --qubit q6 --beta-min -0.6 --beta-max 0 --beta-points 40 --peak-amplitude-v 0.5
python Projects/shaped_pulse_spectroscopy/scripts/optimize_echo_drag_beta.py --dry-run --beta-min -1.5 --beta-max 0 --beta-points 200 --beta-order descending --amplitude-min 0.2 --amplitude-max 1 --amplitude-points 5 --amplitude-spacing linear --frequency-points 200 --num-shots 1000
```

The second example visits beta from 0 down to -1.5 and measures amplitude
prefactors 0.2, 0.4, 0.6, 0.8, and 1.0. Replace `--dry-run` with `--execute`
to acquire. The amplitude count, bounds, and linear/log spacing are configurable.

Each acquisition saves `settings.json`, raw runs, `records.json`, `fits.csv`,
one amplitude-slice fit plot per beta, `beta_summary.png`, and `summary.json` under
`data/echo_drag_beta_1d/<timestamp>/`. `summary.json` fills in `best_beta`
only after the full sweep completes; provisional scores and progress are retained
on interruption. A boundary minimum is flagged. No selected beta is applied
to device profiles automatically.

The two-stage DRAG/AC-Stark workflow fixes `L=20 us` and `cutoff=0.0025` and is
approval gated. First write a review-only coarse plan:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\plan_drag_beta_kappa_calibration.py --target-qubit qX --existing-kappa-mhz-inv KAPPA --output proposed_drag_plan.json
```

The manifest includes the exact target, waveform, beta values, and shared
detuning-amplitude grid plus a SHA-256 approval token. Acquisition accepts only
an exact matching token. The joint grid is generated around the measured best
fixed-kappa beta and has its own manifest/token, so it must be reviewed again.
All runs disable profile updates and proposals. Three-state readout records
measured `P_f` when calibrated G/E/F discrimination is available; selection
then minimizes leakage subject to center-RMS and contrast gates.
Figures include a compact parameter banner with the pulse shape, pulse length,
cutoff or tau, echo flag, peak amplitude, sweep span/step, and square pi pulse.
When available, the banner also includes T1, Ramsey T2*, and
`1 / (pi * T2*)` in Hz.

The MHz conversion is a prediction from the square `x180` calibration, not an
independent measurement. Validate its linearity over the intended voltage range
with the constant-envelope duration-by-amplitude diagnostic:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_rabi_frequency_linearity.py
```

For every constant drive voltage, this experiment fits the observed time-domain
oscillation frequency. Its result compares the measured frequency with the
square-`x180` prediction and plots the relative error. It never updates the
device profile, and amplitudes are checked against the `1 V` waveform bound.
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

To acquire the paper-style six-panel fixed-amplitude comparison at 3, 20, and
40 MHz, with 10,000 shots for each echo/no-echo trace and a matched three-level
simulation, run:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\run_paper_figure3_slices.py
```

The script uses the narrow-map settings (`L=20 us`, `c=0.005`, 501 points over
1 MHz, active reset, and full readout mitigation), saves every hardware run,
and writes PNG/PDF/SVG versions of the 3-by-2 comparison under
`data/paper_figure3_fixed_slices/`. The simulation uses the same vectorized
three-level Lindblad model as the paper figure workflow.

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
python Projects\shaped_pulse_spectroscopy\scripts\run_cutoff_amp_fwhm_map.py
```

This runs ten log-spaced cutoff values from `1e-4` to `0.99`. For each cutoff it
runs the class-based echo-Lorentzian spectroscopy experiment in memory, without
saving the individual per-cutoff experiments. The wrapper writes only the
combined `cutoff_amp_fwhm_map_fit_results.csv`, `cutoff_amp_fwhm_map_best_signal.csv`,
`manifest.json`, `cutoff_amp_fwhm_map_summary.png`, `cutoff_amp_fwhm_map_fwhm_heatmap.png`,
and `cutoff_amp_fwhm_map_per_cutoff_traces.png` under
`data/cutoff_amp_fwhm_map/`. Each completed cutoff also saves only its
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
summary and heatmap are normalized by the qubit's Ramsey T2* FWHM limit
`1 / (pi*T2*)`; for example, `T2*=6.86 us` corresponds to a `46.4 kHz`
normalization unit. New inner datasets and aggregate CSV records save this as
`t2_star_s`, `t2_star_fwhm_limit_hz`, and `fwhm_t2_star_units`.

For long pulses, keep `lorentzian_length_in_ns` as the physical pulse duration
and set `waveform_template_length_in_ns` to a shorter template, for example
`2000` ns. The experiment will store the shorter arbitrary waveform and play it
with QUA `duration=lorentzian_length_in_ns // 4`.

## Robust Cutoff Optimization Roadmap

`cutoff_amp_fwhm_map.py` is a key experiment. It scans the root-Lorentzian
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

- small FWHM in T2*-limit units;
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
cutoff_amp_fwhm_map_fit_results.csv
cutoff_amp_fwhm_map_best_points.csv
cutoff_amp_fwhm_map_quality_summary.csv
cutoff_amp_fwhm_map_rescan_recommendations.csv
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

Use two cutoff regions:

```text
high_cutoff: 0.99 to 0.1, 10 log cutoffs
low_cutoff:  0.1  to 0.01, 10 log cutoffs
```

Use the same three frequency domains for both:

```text
100 MHz, 10 MHz, 1 MHz
```

Default grid:

```text
200 frequency points per domain
100 log-spaced amplitudes
30 averages
echo=false and echo=true
```

Cutoff-region campaign scripts live under:

```text
Projects/shaped_pulse_spectroscopy/scripts/cutoff_scans/
```

Preset region commands:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_high_cutoff.py
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_low_cutoff.py
```

Numbered queue:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\01_high_no_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\02_high_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\03_low_no_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\04_low_echo.py --qubit q1
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

Advice:

- Run no-echo first as the baseline.
- Use `100 MHz` to detect clipping, `10 MHz` for the main comparison, and `1 MHz` only for centered narrow features.
- If the best cutoff differs between echo modes, rerun only the interesting cutoff/domain with more shots.
- Keep the log amplitude minimum positive; the config uses `0.01`.

Existing campaign folders can be summarized or replotted without rerunning:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\build_region_summary.py data\cutoff_amp_fwhm_map\high_cutoff\20260708_153000
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\replot_region.py data\cutoff_amp_fwhm_map\high_cutoff\20260708_153000
```

The region/domain definitions live in:

```text
Projects/shaped_pulse_spectroscopy/configs/cutoff_regions.json
```
