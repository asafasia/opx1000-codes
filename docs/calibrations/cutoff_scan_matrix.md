# Cutoff Amp FWHM Map

Root-Lorentzian `cutoff_amp_fwhm_map` scan plan.

## Grid

| Region | Cutoff Range | Cutoffs | Frequency Domains | Frequency Points | Amplitudes | Averages | Echo |
| --- | --- | ---: | --- | ---: | --- | ---: | --- |
| `high_cutoff` | `0.99` to `0.1` | 10 log | `100`, `10`, `1 MHz` | 200/domain | 100 log | 30 | false, true |
| `low_cutoff` | `0.1` to `0.01` | 10 log | `100`, `10`, `1 MHz` | 200/domain | 100 log | 30 | false, true |

Config:

```text
Projects/shaped_pulse_spectroscopy/configs/cutoff_regions.json
```

## Run Order

These commands only print the plan unless `--execute` is added.

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\01_high_no_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\02_high_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\03_low_no_echo.py --qubit q1
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\04_low_echo.py --qubit q1
```

Run one domain only:

```powershell
python Projects\shaped_pulse_spectroscopy\scripts\cutoff_scans\run_domain.py --region high_cutoff --domain domain_10mhz --qubit q1
```

## Advice

- Start without echo first; it is the cleaner baseline.
- Keep `100 MHz` as the clipping guard, not the linewidth estimate.
- Trust `10 MHz` for broad-to-medium features.
- Trust `1 MHz` only when the feature is centered and not edge-clipped.
- If the best cutoff changes between echo modes, rerun only the winning area with more shots.
- If runtime is too long, reduce domains first, not cutoff count.
- For log amplitudes, keep the minimum amplitude positive; this setup uses `0.01`.
