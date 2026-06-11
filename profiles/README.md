# Device Profiles

Profiles separate hardware connectivity, calibrated device parameters, and
pulse definitions. Each profile is a versioned directory:

```text
profiles/
  main/
    profile.json       Profile manifest and active qubits
    connectivity.json Hardware, network, ports, line connections, and LOs
    qubits.json        Qubit, resonator, coherence, and readout parameters
    pulses.json        Reusable pulse definitions
```

All physical values include their unit in the field name, such as
`frequency_hz`, `length_ns`, and `axis_angle_rad`. This avoids implicit-unit
mistakes and keeps the JSON readable without a custom parser.

Pulse definitions are reusable. Qubits reference them by name under
`operations`, so a calibrated pulse can be shared or replaced without copying
its parameters. Supported pulse types are:

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
q1 = profile["qubits"]["qubits"]["q1"]
```

Generate the QuAM wiring and base state from the main profile:

```powershell
python -m quam_config.wiring_lffem_mwfem
```

Then apply the profile's calibrated qubit, resonator, port, and pulse values:

```powershell
python -m quam_config.populate_quam_lf_mw_fems --profile main
```

Test profile application without writing `state.json`:

```powershell
python -m quam_config.populate_quam_lf_mw_fems --profile main --no-save
```

The preferred single-step command creates the wiring, base QuAM, calibrated
parameters, and pulses together:

```powershell
python -m quam_config.create_machine_from_profile --profile main
```

Create and validate the complete machine without writing files:

```powershell
python -m quam_config.create_machine_from_profile --profile main --no-save
```

Use it from Python:

```python
from quam_config.create_machine_from_profile import create_machine_from_profile

machine = create_machine_from_profile("main")
```

Calibration experiments can use the shorter in-memory factory:

```python
from quam_config import create_machine

machine = create_machine()          # profiles/main
machine = create_machine("testing") # profiles/testing
```

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
