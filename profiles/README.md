# Device Profiles

Profiles separate hardware connectivity, calibrated device parameters, and
pulse definitions. Each profile is a versioned directory:

```text
profiles/
  main/
    profile.json       Profile manifest and the single active-qubit list
    connectivity.json Hardware, network, ports, line connections, and LOs
    qubits.json        Qubit, resonator, coherence, and readout parameters
    pulses.json        Reusable pulse definitions
```

All physical values include their unit in the field name, such as
`frequency_hz`, `length_ns`, and `axis_angle_rad`. This avoids implicit-unit
mistakes and keeps the JSON readable without a custom parser.

Set each qubit's `transmon.thermalization_time_ns` explicitly. Calibration
experiments use this value through `qubit.reset_qubit_thermal()`. Because QuAM
represents thermalization as an integer multiple of T1, the configured value
must be an integer multiple of `t1_ns`, or of 10,000 ns when T1 is unknown.

Readout acquisition timing is configured per qubit with
`readout.time_of_flight_ns`, `readout.smearing_ns`, and
`readout.depletion_time_ns`. The profile population step applies these values
directly to the QuAM resonator.

Pulse definitions are grouped by qubit name in `pulses.json`. Each qubit
references pulse names from its own group under `operations`, so the same
operation names can be calibrated independently. Supported pulse types are:

- `constant`: rectangular envelope; the only allowed resonator/readout type.
- `drag`: requires `sigma_ns`, `beta`, and `detuning_hz`.
- `cosine`: cosine-shaped qubit-control envelope.
- `saturation`: long constant qubit drive.

Validate the main profile:

```powershell
python -m profiles.validate_profile main
```

Load it from Python:

```python
from profiles import load_profile

profile = load_profile("main")
q9 = profile["qubits"]["qubits"]["q9"]
```

Create and validate the complete machine in memory:

```powershell
python -m quam_config.create_machine_from_profile --profile main --no-save
```

Define activation only in `profile.json.active_qubits`. Qubit entries in
`qubits.json` contain parameters for both active and inactive qubits and must
not define an `enabled` field.

Use it from Python:

```python
from quam_config.create_machine_from_profile import create_machine_from_profile

machine = create_machine_from_profile("main", save=False)
```

Calibration experiments use the shorter in-memory factory at startup. Generated
`state.json`, `wiring.json`, and physical-state files are not repository inputs:

```python
from quam_config import create_machine

machine = create_machine()          # profiles/main
machine = create_machine("testing") # profiles/testing
```

Qualibrate still stores an explicit machine snapshot with each saved run, but
implicit saves to the repository root are disabled.

Set `QUAM_PROFILE` to select a profile for all calibration experiments without
editing their source:

```powershell
$env:QUAM_PROFILE = "main"
python calibrations/03a_qubit_spectroscopy.py
```

Select another profile or skip the wiring plot with:

```powershell
python -m quam_config.wiring_lffem_mwfem --profile main --no-plot
```

The wiring builder uses the profile's network settings, MW-FEM inventory,
active qubits, exact XY ports, and shared resonator feedlines. Importing the
module does not write configuration files; generation happens only when the
module is run.

## Recommended Workflow

Keep `main` as the currently trusted calibration. Create a new profile folder
for experiments or cooldowns, validate it, and promote it to `main` only after
testing. For larger systems, the next useful addition is JSON Schema files and
applying the profile's calibrated qubit and pulse values to the generated QuAM
state.
