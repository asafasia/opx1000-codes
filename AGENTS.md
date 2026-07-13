# Agent Notes

## Project Overview

This repository contains OPX1000 quantum-control calibration code. Current
calibration work should generally use `calibrations/`, not
`calibrations_old/`.

Generated experiment output goes under `data/` and should not be committed
unless explicitly requested.

## Environment

Use the lab Python environment that contains the OPX/QUA, Qualibrate, QuAM,
NumPy, Matplotlib, xarray, and pytest dependencies. From the repository root,
make the package importable when running scripts directly:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

The common Python environment path used in this workspace is:

```powershell
C:\Users\owner\miniconda3\envs\opx1000_env\python.exe
```

## Important Commands

Validate profiles:

```powershell
python -m profiles.validate_profile main
python -m profiles.validate_profile single_qubit --qubit q3
```

Build a machine in memory without writing generated QuAM files:

```powershell
python -m quam_config.create_machine_from_profile --profile main --no-save
python -m quam_config.create_machine_from_profile --profile single_qubit --qubit q3 --no-save
```

Run tests:

```powershell
python -m pytest
```

Use the calibration runner:

```powershell
python -m calibrations.runner list
python -m calibrations.runner describe resonator
python -m calibrations.runner run power-rabi --qubit q9 --simulate --no-save
```

Use `--dry-run`, `--simulate`, or `--no-save` when checking behavior without
intending to run or persist a real calibration.

## Calibration Conventions

New calibration code should subclass `BaseCalibration` from
`calibrations/base.py`.

Use `CalibrationOptions` to avoid unwanted side effects in automated runs:

```python
CalibrationOptions(
    save_raw_data=False,
    save_figures=False,
    plot_data=False,
    update_state=False,
    propose_profile_update=False,
    apply_profile_update=False,
)
```

Profile updates should be staged or proposed by default. Do not apply profile
updates unless the user explicitly asks.

## Profile Safety

`profiles/` is the source of truth for executable device configuration.

Respect unit-suffixed field names such as `_hz`, `_ns`, and `_rad`.

Pulse amplitudes are limited to absolute value `0.7`. Do not raise this limit
without explicit user approval and a hardware-safety reason.

Generated root-level QuAM files such as `state.json` and `wiring.json` are not
repository inputs.

## Hardware Safety

Avoid running real hardware calibrations unless the user explicitly asks. Prefer
`--simulate`, `--dry-run`, or `--no-save` when validating code behavior.

Do not change frequencies, amplitudes, integration kernels, LO settings, or
wiring/connectivity casually. Explain any proposed hardware-impacting change.

## Data And Git Hygiene

Do not commit generated files from `data/`, cache files, figures, or local app
logs unless requested.

Be careful with existing modified profile files under `profiles/`; they may
contain recent calibration results.

Before edits, check:

```powershell
git status --short
```

## Where To Look

- `README.md`: high-level repository overview.
- `calibrations/README.md`: calibration lifecycle and runner usage.
- `profiles/README.md`: profile schema and safety constraints.
- `docs/hardware/`: durable hardware facts.
- `tests/`: expected validation patterns.
