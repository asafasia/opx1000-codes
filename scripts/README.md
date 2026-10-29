# Resonator spectroscopy versus qubit drive frequency

`resonator_vs_qubit_frequency.py` ports the outer sweep and summary plots from
`opx1000_Or/scripts/resonator_vs_qubit_frequency.py` to this project's profiles
and `calibrations/02a_resonator_spectroscopy.py` lifecycle.

From the repository root, with the lab Python environment active:

```powershell
python scripts/resonator_vs_qubit_frequency.py --profile Or_profile --qubit q3b --dry-run
```

Remove `--dry-run` to acquire on hardware. `--no-show` saves plots without opening
windows. Use the current profile qubit names, such as `q1b`, `q2b`, and `q3b`.

The default sweep has 100 physical qubit RF frequencies from 3.5 to 5.5 GHz,
including endpoints. Each point sets drive LO = RF - 100 MHz in an independent
in-memory profile, leaving readout settings and pulse amplitudes from the
selected profile. Every profile, generated configuration and QUA program is
checked before acquisition starts. Profile files are never updated.

Customize the sweep with `--start-hz`, `--stop-hz`, `--points`, and `--if-hz`.
Readout scanning defaults to 200 shots, 30 MHz span, and 0.3 MHz steps; use
`--num-shots`, `--readout-span-mhz`, and `--readout-step-mhz` to override these.
Use `--readout-amplitude-factor 0.9` to multiply the selected readout pulse's
amplitude in memory. This leaves the drive pulse and profile files unchanged;
each run's manifest records the original amplitude, factor, and effective amplitude.
Only profiles with `build_mode: single_qubit` are supported.

This uses the current calibration's thermal reset and continuous saturation
during readout, rather than the source project's saturation-before-readout
sequence. The profile saturation pulse must cover the lead-in and readout;
the existing calibration validates this. The two responses are drive off and
saturated, not calibrated pure ground/excited states. The largest IQ separation
is a sampled contrast score, not assignment fidelity or a fitted qubit resonance.

Results go to `data/resonator_vs_qubit_frequency/<timestamp>_<qubit>/` by default
(`--output-dir` changes the root). Each point saves its effective profile,
parameters, raw shot arrays and their xarray metadata, and response statistics.
`manifest.json` checkpoints completed points and records failures/interruption.
Completed sweeps also produce `summary.npz`, `summary.png`, and pages of
resonator traces. Dry runs connect to no hardware and save no experiment output.
