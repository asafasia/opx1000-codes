# Echo Lorentizan

This project is a frequency-versus-amplitude 2D sweep that mirrors the power
Rabi chevron, but the qubit drive operation is a user-length Lorentzian
waveform instead of a square, cosine, or DRAG pulse.

The pulse envelope is

```text
A / (1 + (t / tau)^2)
```

where `t` is centered on the midpoint of the pulse. The key parameters are:

- `lorentzian_length_in_ns`: total waveform length.
- `lorentzian_tau_in_ns`: Lorentzian width parameter.
- `lorentzian_peak_amplitude`: unscaled peak amplitude `A`.
- `min_amp_factor`, `max_amp_factor`, `amp_factor_step`: y-axis amplitude sweep.
- `frequency_span_in_mhz`, `frequency_step_in_mhz`: x-axis detuning sweep.

Run the sweep with:

```powershell
python Projects\echo-lorentizan\echo_lorentzian_sweep.py
```
