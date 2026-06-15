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
calibrations/          Qualibrate calibration and characterization scripts
calibration_utils/     Per-calibration parameters, analysis, and plotting code
profiles/              Versioned JSON device profiles and validation helpers
profiles/profile_updater.py
                       Profile update staging/apply helper
quam_config/           QuAM machine construction from profiles
docs/                  Hardware and repository documentation
apps/visualiser/       Read-only local dashboard for saved experiment data
apps/profile_studio/   Local structured editor for profile JSON files
calibration_io/        Calibration result persistence helpers
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
python calibrations/03a_qubit_spectroscopy.py
```

## Running Calibrations

Calibration scripts live in `calibrations/` and are organized roughly in the
order they are used during bring-up:

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
12_Qubit_Spectroscopy_ef.py
13_power_rabi_ef.py
14_gef_readout_frequency_optimization.py
```

Most scripts are Qualibrate nodes. They build a machine from the selected
profile, run or simulate a QUA program, analyze the result with the matching
module under `calibration_utils/`, and save outputs under `data/`.

## Local Tools

Start the read-only experiment browser:

```powershell
python apps/visualiser/server.py
```

Open <http://127.0.0.1:8765>.

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
