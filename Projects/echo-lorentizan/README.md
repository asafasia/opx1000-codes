# Echo Lorentizan

This project is a frequency-versus-amplitude 2D sweep that mirrors the power
Rabi chevron, but the qubit drive operation is a user-length Lorentzian-like
waveform instead of a square, cosine, or DRAG pulse.

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
detuning and overlays the fitted FWHM edges as paired markers on the heatmap.

Run the sweep with:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_sweep.py
```

The class-based v2 version lives in the same project folder and can be run with:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_v2.py
```

The minimalist amplitude-only version keeps detuning at zero and sweeps only
the Lorentzian amplitude:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_amplitude_v2.py
```

For long pulses, keep `lorentzian_length_in_ns` as the physical pulse duration
and set `waveform_template_length_in_ns` to a shorter template, for example
`2000` ns. The experiment will store the shorter arbitrary waveform and play it
with QUA `duration=lorentzian_length_in_ns // 4`.
