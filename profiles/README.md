# Device Profiles

Profiles separate hardware connectivity, calibrated device parameters, and
pulse definitions. Each profile is a versioned directory:

Physical wiring constraints and vendor reference data are documented under
[`docs/hardware`](../docs/hardware/). Profiles are the executable configuration
and must respect those hardware constraints.

```text
profiles/
  main/
    profile.json       Profile manifest and the single active-qubit list
    connectivity.json Hardware, network, ports, line connections, and LOs
    qubits.json        Qubit, resonator, coherence, and readout parameters
    pulses.json        Reusable pulse definitions
  single_qubit/
    profile.json       Independent single-qubit profile manifest
    connectivity.json Independent hardware, port, and LO configuration
    qubits.json        Independent qubit and readout parameters
    pulses.json        Independent control and readout pulse definitions
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

Readout pulses define piecewise-constant integration kernels as
`integration_weights: [[weight, length_ns], ...]`. The segment lengths must
span the full readout pulse. The per-qubit
`readout.integration_weights_angle_rad` rotates this kernel when the QuAM
configuration is generated.

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
machine = create_machine(qubit="q3") # profiles/single_qubit, q3 only
```

`create_machine(qubit="q3")` builds a machine containing only `q3`, its
resonator, its drive line, and the controller/FEM ports those objects use. The
single-qubit machine uses only the independent files under
`profiles/single_qubit`. The selected qubit is isolated before wiring is built.
Nothing is copied or inferred from `main` at load time, so its LOs,
frequencies, pulses, amplitudes, and readout parameters can be calibrated
independently.

Per-qubit drive and readout LOs are stored under each connection's
`lo_frequencies_hz` in `profiles/single_qubit/connectivity.json`. They override
the selected physical output ports only after one qubit has been selected.

The factory class is also available directly:

```python
from quam_config import CreateMachine

cm = CreateMachine()            # profiles/main
machine = cm.machine
cm.profile.save()

cm = CreateMachine(qubit="q3")  # profiles/single_qubit, q3 only
machine = cm.machine
cm.profile.save()               # saves the loaded full profile documents
```

## CreateMachine, Profile Loading, And Saving

`CreateMachine` is the normal Python entry point for building an in-memory QuAM
machine from the repository profiles. It does not read root-level `state.json`
or `wiring.json`; instead, it loads JSON from `profiles/`, validates it, builds
the wiring, creates a fresh QuAM object, and applies the profile values.

The short helper:

```python
from quam_config import create_machine

machine = create_machine()
```

is equivalent to:

```python
from quam_config import CreateMachine

cm = CreateMachine()
machine = cm.machine
```

Use `CreateMachine` directly when you also need access to the profile object
that was used to build the machine:

```python
cm = CreateMachine()
machine = cm.machine
profile = cm.profile
```

`cm.profile` is the repository profile object. It is not a live reverse mapping
from the QuAM object back into JSON. If code changes `machine` directly, those
changes are not automatically written into `profile.documents`; the profile
documents must be updated deliberately before calling `profile.save()`.

The `CreateMachine` object forwards unknown attributes to the machine, so code
that receives `cm` can often use it like the machine itself. Prefer
`cm.machine` when the distinction matters.

### Profile Selection Rules

By default, `CreateMachine()` uses `profiles/main`:

```python
cm = CreateMachine()
```

An explicit profile can be selected by name:

```python
cm = CreateMachine("main")
cm = CreateMachine(profile_name="main")
cm = CreateMachine(mode="main")
```

Single-qubit work uses the independent `profiles/single_qubit` profile. Passing
only a qubit name selects that profile automatically:

```python
cm = CreateMachine(qubit="q3")
```

This builds a machine containing only the selected qubit, its resonator, its
drive line, and the controller/FEM ports needed by that qubit. It does not copy
values from `profiles/main`.

If a qubit is provided together with an explicit profile, that profile must
support single-qubit selection:

```python
cm = CreateMachine("single_qubit", qubit="q3")
```

Selecting a qubit from `main` is rejected because `main` is treated as the
full-chip profile.

### Profile.load()

`Profile.load()` reads the four JSON documents from the profile directory:

- `profile.json`
- `connectivity.json`
- `qubits.json`
- `pulses.json`

It validates the schema version, pulse definitions, qubit parameters,
connectivity, active-qubit list, MW-FEM bands, LO ranges, and operation
references. If validation fails, it raises `ProfileError`.

Example:

```python
from profiles import Profile

profile = Profile("main")
documents = profile.load()
```

The loaded documents are normal Python dictionaries. `profile.documents` keeps
a deep copy of the full loaded profile.

For `single_qubit`, loading with a qubit selection returns a projected profile
containing only that qubit:

```python
profile = Profile("single_qubit", qubit="q3")
documents = profile.load()
```

That selected projection is useful for building a one-qubit machine, but it is
not a full profile directory.

### Profile.save()

`Profile.save()` writes validated profile dictionaries back to the JSON files
inside the profile directory. It writes through temporary files and replaces the
targets, so a save should not leave partially written JSON files.

Save the currently loaded profile:

```python
profile = Profile("main")
documents = profile.load()
documents["qubits"]["qubits"]["q3"]["frequencies_hz"]["qubit_f01"] = 4.5e9
profile.save(documents)
```

Or save `profile.documents` after it has been loaded and modified:

```python
profile = Profile("main")
profile.load()
profile.documents["manifest"]["active_qubits"] = ["q3"]
profile.save()
```

`Profile.save()` saves profile documents, not arbitrary changes made to a built
QuAM machine. Calibration code that computes a new frequency, amplitude,
threshold, or pulse value must write that value into the appropriate dictionary
inside the profile documents before saving.

`Profile.save()` refuses to save a selected single-qubit projection over the
full `single_qubit` profile. To edit the full `single_qubit` profile, load it
without selecting a qubit, modify the full documents, and then save.

### QuAM Machine Save

Profile saving is different from QuAM machine saving.

`create_machine_from_profile(..., save=True)` calls `machine.save()` after the
QuAM object is built. That writes generated QuAM artifacts such as `state.json`
and `wiring.json`.

`CreateMachine` intentionally calls `create_machine_from_profile(...,
save=False)`. This keeps those generated artifacts out of the repository root
during normal calibration startup. The calibration node still saves its own
run-specific snapshot with the experiment results.

Use profile JSON as the durable source of truth. Treat generated QuAM files as
outputs, not as the hand-edited calibration input.

Validate or build a selected single-qubit profile:

```powershell
python -m profiles.validate_profile single_qubit --qubit q3
python -m quam_config.create_machine_from_profile --profile single_qubit --qubit q3 --no-save
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
all configured qubits, exact XY ports, and shared resonator feedlines.
`active_qubits` controls which qubits calibrations select by default, without
removing inactive qubits from the physical wiring. Importing the module does
not write configuration files; generation happens only when the module is run.

## Recommended Workflow

Keep `main` as the currently trusted calibration. Create a new profile folder
for experiments or cooldowns, validate it, and promote it to `main` only after
testing. For larger systems, the next useful addition is JSON Schema files and
applying the profile's calibrated qubit and pulse values to the generated QuAM
state.

When several users are calibrating, profile edits should be isolated by git
branch until the intended values are reviewed. Frequencies, amplitudes,
thresholds, pulse definitions, and readout weights are shared calibration state;
editing them directly on the same branch can overwrite another user's working
calibration. Treat each branch as one coherent profile workspace, then merge or
promote values deliberately.

For easier manual edits, use Profile Studio:

```powershell
python profile_studio/server.py
```

Then open <http://127.0.0.1:8766>. The HTML editor exposes the profile files as
Profile, Qubits, Pulses, and Connectivity tabs and saves changes back to the
selected profile JSON.
