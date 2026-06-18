# OPX1000 Codes

Calibration, configuration, and experiment utilities for an OPX1000-based
quantum-control setup. The repository combines Qualibrate/QUA calibration
nodes, QuAM machine construction from JSON device profiles, analysis helpers,
classical Ising experiments, and local tools for reviewing saved data.

The durable hardware facts live under `docs/`; the executable device
configuration lives under `profiles/`; generated experiment output is written
under `data/` and is intentionally not tracked by git.

## Repository Layout

```text
calibrations_v2/       Class-based Qualibrate calibration scripts and base lifecycle
calibrations_old/      Older script-style calibration implementations
calibration_utils/     Per-calibration parameters, analysis, and plotting code
profiles/              Versioned JSON device profiles and validation helpers
profiles/profile_updater.py
                       Profile update staging/apply helper
quam_config/           QuAM machine construction from profiles
docs/                  Hardware and repository documentation
apps/visualiser/       Read-only local dashboard for saved experiment data
apps/parameter_scan/   Local live control and monitoring app for long scans
apps/profile_studio/   Local structured editor for profile JSON files
calibration_io/        Calibration result persistence helpers
workflows/             Multi-step calibration workflows
sweeps/                Higher-level sweep runners around calibration workflows
Projects/ising_machine/
                       QUA-based classical/pseudo-spin Ising experiments
Projects/ising_simulation/
                       Pure-Python 2D Ising Monte Carlo simulation
Projects/dynamic_circuit_active_reset/
                       Separate active-reset experiment
tests/                 Unit tests for profiles, calibration analysis, and QUA logic
utils/                 Shared simulation, plotting, and readout helpers
```

More focused documentation is available in:

- `docs/README.md`
- `docs/hardware/README.md`
- `profiles/README.md`
- `calibrations_v2/README.md`
- `apps/visualiser/README.md`
- `apps/profile_studio/README.md`
- `Projects/ising_machine/README.md`
- `Projects/ising_simulation/README.md`
- `Projects/dynamic_circuit_active_reset/README.md`

## Environment

Use the lab Python environment that contains the OPX/QUA, Qualibrate, QuAM,
NumPy, Matplotlib, xarray, and pytest dependencies. From the repository root,
make the package importable when running scripts directly:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

For the standalone Ising simulation only, the minimal dependencies are listed
in `Projects/ising_simulation/requirements.txt`:

```powershell
python -m pip install -r Projects/ising_simulation/requirements.txt
```

## Device Profiles

The profile system is the source of truth for machine construction. The main
profile is `profiles/main`; isolated single-qubit work uses
`profiles/single_qubit`.

Validate the main profile:

```powershell
python -m profiles.validate_profile main
```

Validate a selected single-qubit profile:

```powershell
python -m profiles.validate_profile single_qubit --qubit q3
```

Build a machine in memory without writing generated QuAM files:

```powershell
python -m quam_config.create_machine_from_profile --profile main --no-save
python -m quam_config.create_machine_from_profile --profile single_qubit --qubit q3 --no-save
```

Calibration scripts call `quam_config.create_machine()` and default to
`profiles/main`. Set `QUAM_PROFILE` to select another profile without editing
the calibration source:

```powershell
$env:QUAM_PROFILE = "main"
python calibrations_v2/03a_qubit_spectroscopy.py
```

## Running Calibrations

Current calibration scripts live in `calibrations_v2/`. The v2 scripts share a
class-based lifecycle in `calibrations_v2/base.py`: create or simulate the QUA
program, save raw xarray results with a profile snapshot, reload and analyze
saved runs, save figures, stage optional profile updates, and clean up temporary
machine changes. Older script-style versions are kept under `calibrations_old/`
for reference.

The scripts are organized roughly in the order they are used during bring-up:

```text
00_hello_qua.py
01b_time_of_flight_mw_fem.py
02a_resonator_spectroscopy.py
02b_resonator_spectroscopy_vs_power.py
03a_qubit_spectroscopy.py
04*_rabi_and_power_rabi.py
05_T1.py
06a_ramsey.py
07*_iq_blobs.py
08*_readout_optimization.py
10b_drag_calibration_180_minus_180.py
11a_single_qubit_randomized_benchmarking.py
11b_single_qubit_randomized_benchmarking_interleaved.py
12_Qubit_Spectroscopy_ef.py
13_power_rabi_ef.py
14_gef_readout_frequency_optimization.py
15_iq_blobs_gef.py
```

Most scripts are Qualibrate nodes. They build a machine from the selected
profile, run or simulate a QUA program, analyze the result with the matching
module under `calibration_utils/`, and save outputs under `data/`.

Runtime side effects can be controlled with `CalibrationOptions`, which lets
you disable raw saves, figures, plotting, state updates, or profile-update
proposals when running unattended scans or nested workflows:

```python
from calibrations_v2 import CalibrationOptions

options = CalibrationOptions(
    save_raw_data=False,
    save_figures=False,
    plot_data=False,
    update_state=False,
    propose_profile_update=False,
    apply_profile_update=False,
)
```

Pass `options=options` into any v2 calibration constructor. See
`calibrations_v2/README.md` for the full subclassing pattern and a concrete
`PowerRabi` example.

## Gate Tune-Up Workflows

The `workflows/` and `sweeps/` packages provide higher-level routines that
reuse the v2 calibrations without writing intermediate profile files.

Run the standard single-qubit DRAG tune-up sequence: Power Rabi, DRAG
180/-180, then single-qubit randomized benchmarking:

```powershell
python workflows/drag_workflow.py
```

Sweep DRAG beta values and validate each point with randomized benchmarking:

```powershell
python sweeps/drag_sweep.py
```

Sweep pulse gate length and run the full DRAG workflow at each valid 4 ns
length:

```powershell
python sweeps/gate_length_drag_workflow_sweep.py
```

These runners save compact aggregate summaries and figures through
`calibration_io.CalibrationSaver`, and they preserve partial results when a
long run is interrupted.

## Overnight Parameter Scans

Use the parameter scan runner for long unattended drift checks. It runs existing
calibration scripts in loops, extracts only fitted values such as T1, Ramsey T2,
frequency offsets, qubit frequency, and linewidth, and writes compact summaries
under `data/parameter_scans/`.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m parameter_scans --config parameter_scans/example_scan.json
```

Run selected experiments without a config file:

```powershell
python -m parameter_scans --name weekend_drift --repetitions 48 --interval-seconds 600 `
  --experiment calibrations_v2/03a_qubit_spectroscopy.py `
  --experiment calibrations_v2/05_T1.py `
  --experiment calibrations_v2/06a_ramsey.py
```

By default the runner stops on the first script error, records the traceback in
`events.jsonl`, and keeps the partial `summary.csv`/`summary.json` already
written. Create a `STOP` file inside the active scan run directory to request a
clean stop after the current experiment. Long scans suppress full raw
calibration saves by default; pass `--save-full-results` only when you want the
usual per-experiment raw data and figures as well.

## Local Tools

Start the read-only experiment browser:

```powershell
python apps/visualiser/server.py
```

Open <http://127.0.0.1:8765>.

Start the live parameter-scan control app:

```powershell
python apps/parameter_scan/server.py
```

Open <http://127.0.0.1:8770>. The app lets you choose calibration scripts,
start/stop a long scan, and watch fitted parameters, variance, drift per hour,
and stability flags update while the run is still active.

Start the profile editor:

```powershell
python apps/profile_studio/server.py
```

Open <http://127.0.0.1:8766>.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest
```

Run a focused test file:

```powershell
python -m pytest tests/test_profile.py
```

## Generated Files

The following are generated during runs and are excluded from version control:

- `data/`
- root-level generated QuAM artifacts such as `state.json`, `wiring.json`,
  `physical_state.json`, and `physical_config.json`
- Python and test caches such as `__pycache__/` and `.pytest_cache/`

Keep durable setup notes in `docs/`, keep executable configuration in
`profiles/`, and keep analysis logic in `calibration_utils/` so calibration
runs remain reproducible.
