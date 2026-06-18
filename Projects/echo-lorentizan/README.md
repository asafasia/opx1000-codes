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
tau = t_cut / sqrt(1 / root_lorentzian_cutoff^2 - 1)
```

For both pulse shapes, `t` is centered on the midpoint of the pulse. The key
parameters are:

- `pulse_shape`: either `lorentzian` or `root_lorentzian`.
- `lorentzian_length_in_ns`: total waveform length.
- `lorentzian_tau_in_ns`: standard Lorentzian width parameter.
- `root_lorentzian_cutoff`: root-Lorentzian edge/peak amplitude ratio.
- `echo`: when `True`, multiply the waveform by a midpoint sign flip so the
  first half is positive and the second half is negative.
- `lorentzian_peak_amplitude`: unscaled peak amplitude `A`.
- `min_amp_factor`, `max_amp_factor`, `amp_factor_step`: y-axis amplitude sweep.
- `frequency_span_in_mhz`, `frequency_step_in_mhz`: x-axis detuning sweep.

The plot shows detuning on the lower x-axis and absolute RF frequency on the
upper x-axis. Its left y-axis shows the equivalent Rabi frequency in Hz,
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

Run the sweep with:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_sweep.py
```

The class-based v2 version lives in the same project folder and can be run with:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_v2.py
```
