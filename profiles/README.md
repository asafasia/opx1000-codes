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

T1 runs with `initial_state="e"` retain the standard `coherence.t1_ns`
metric and `qubit.T1`. Runs with `initial_state="g"` save their fitted rise
constant separately as `T1_ge`, using `coherence.t1_ge_ns` in `metrics.json`
and `qubit.extras["T1_ge"]` in seconds in QuAM snapshots. This optional metric
does not affect the standard T1 or thermalization timing.

Readout acquisition timing is configured per qubit with
`readout.time_of_flight_ns`, `readout.smearing_ns`, and
`readout.depletion_time_ns`. The profile population step applies these values
directly to the QuAM resonator.

MW-FEM acquisition gain is configured per physical input as
`controllers.<controller>.fems.<fem>.inputs.<port>.gain_db` in
`connectivity.json`. Valid values are integer dB settings from 0 through 32;
all resonators sharing that physical input use the same gain.

The profile-level `readout_discriminator` selects the QUA state classifier for
all qubits. `"quam"` selects binary I-threshold discrimination and is the default.
`"nearest_center"` selects the repository macro, which assigns the measured IQ
point to the nearest calibrated blob center. The optional per-qubit
`readout.gef_centers` contains either the G/E centers (2x2) or G/E/F centers
(3x2), in demodulation units. IQ-blobs calibration stages these values.

Experiments can use `readout_state_configured(...)` for one measurement or
`active_reset_configured(...)` for bounded active reset. Both macros select the
profile discriminator automatically. Active reset supports G/E and G/E/F; the
three-state path applies E-to-G or F-to-E-to-G correction pulses according to
the discriminated state.

Pulse definitions are grouped by qubit name in `pulses.json`. Each qubit
references pulse names from its own group under `operations`, so the same
operation names can be calibrated independently. Supported pulse types are:

- `constant`: rectangular envelope for qubit control or readout.
- `flat_top_gaussian`: readout envelope with Gaussian edges; `edge_length_ns` defaults to 100 ns.
- `drag`: requires `sigma_ns`, `beta`, and `detuning_hz`.
- `cosine`: cosine-shaped qubit-control envelope.
- `saturation`: long constant qubit drive.

Profile pulse amplitudes are limited to an absolute value of `1.0` for all
pulse types. Values above this limit are rejected during profile validation with
a `ProfileError` before the QuAM machine is built. This protects the machine
builder from applying unsafe profile values. If a calibration proposes a larger
amplitude, review the profile, full-scale power, and calibration result before
changing the limit.

Select the readout pulses in `profile.json`, next to `readout_discriminator`:

```json
"readout_discriminator": "nearest_center",
"readout_pulse": "readout_flattop",
"readout_gef_pulse": "readout_GEF"
```

- `readout_pulse`: `"readout"` (square) or `"readout_flattop"` (Gaussian edges).
- `readout_gef_pulse`: `"readout_GEF"` (square) or `"readout_GEF_flattop"` (Gaussian edges).

These choices apply to every selected qubit in that profile. Experiment operation
names remain `readout` and `readout_GEF`; rebuilding the machine applies the
selection automatically. Profiles without these selectors retain their existing
per-qubit operation mappings.

Each shape has a separate entry under `pulses.json.pulses.<qubit>`, with its own
amplitude and length. For example, `readout_flattop` contains:

```json
{
  "target": "resonator",
  "type": "flat_top_gaussian",
  "amplitude": 0.1,
  "length_ns": 2000,
  "edge_length_ns": 100,
  "digital_marker": "ON"
}
```

Amplitude calibration proposals target the selected pulse definition. GE and GEF
retain their respective readout settings and frequencies. Optimized kernel files
remain keyed by the experiment operation (`readout` or `readout_GEF`); switching
shape does not select a separate stored IQ calibration or kernel. Recalibrate
those for the newly selected shape before using discrimination or optimized weights.

`length_ns` includes both edges: this example has a 100 ns rise, an 1800 ns
plateau, and a 100 ns fall. Gaussian sigma is `edge_length_ns / 5`.
The amplitude is the plateau amplitude; no area normalization is applied.
Edge and total durations must be multiples of 4 ns, with a positive plateau;
total duration must be at least 16 ns. The Gaussian readout amplitude is capped
at 0.7. Each readout mode selects its shape independently.

Changing shape or edge duration changes the IQ response. Recalibrate IQ blobs
with thermal reset and, if used, optimized integration weights before relying
on the new pulse for discrimination. Recorded IQ calibration signatures include
the Gaussian shape and edge duration. Use a square pulse for time-of-flight
calibration; the MW-FEM diagnostic can override either profile shape locally.

Readout pulse entries do not need `integration_weights`. With `use_kernel: false`,
QuAM automatically uses constant unit weights across the full pulse length,
including both Gaussian edges when selected. The weights also follow changes to
`pulse.length` in memory. The per-qubit `integration_weights_angle_rad` still
rotates the weights. Legacy inline `integration_weights` fields are ignored.
When assigning custom weights directly to a QuAM pulse, clear its automatic
reference first (`pulse.integration_weights = None`), then assign the segments.

The optional `readout.confusion_matrix` is the 2x2 assignment matrix produced
by binary IQ-blobs analysis. Rows identify prepared `g/e` states and columns
identify discriminated `g/e` states. It is loaded onto the resonator for
readout-error mitigation.

Set `qubits.json.qubits.<qubit>.readout.use_kernel` for GE readout, or
`readout_gef.use_kernel` for GEF readout:

- `false`: automatically use constant integration weights spanning the pulse.
- `true`: load the optimized `profile_kernel` and `time_ns` arrays from
  `profiles/<profile>/kernels/<qubit>_<pulse_name>_kernel.npz`
  (normally `<qubit>_readout_kernel.npz` or `<qubit>_readout_GEF_kernel.npz`).
  The saved kernel must span the selected pulse length; missing or mismatched
  kernels produce an error.

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

Coherence values (`T1`, Ramsey `T2*`, and echo `T2`) have one operational source
of truth: `metrics.json` under `qubits.<name>.coherence`. Machine construction
copies those metrics into QuAM's `T1`, `T2ramsey`, and `T2echo` fields. The
similarly named values under `qubits.json.transmon` are read only as a legacy
fallback for profiles without a metrics document.

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

## External DC Bias

The optional `connectivity.dc_bias` object configures the host-side Arduino
DC-bias QuAM component:

```json
"dc_bias": {
  "port": "COM7",
  "baud_rate": 115200,
  "channel_count": 8,
  "max_abs_voltage_v": 0.01
}
```

`connectivity.dc_bias.max_abs_voltage_v` in the selected profile is the single
source of the voltage limit. It must be finite and positive; qubit bias values,
external-bias sweeps, and the Arduino driver all enforce this same value.
There is no separate hardcoded ceiling. Building a
machine attaches these settings as `machine.dc_bias` but does not open the
serial port or change an output. Hardware access occurs only when host code
calls methods such as `machine.dc_bias.set_voltage(...)` or enters
`machine.dc_bias.applied(...)`.

In the `single_qubit` profile, each qubit stores its bias voltage using a
unit-suffixed field:

```json
"q3": {
  "dc_bias_v": 0.0
}
```

The hardware output is deliberately not part of the profile: channel 0 (physical output 1) is
hardcoded in `ArduinoDCBias` and shared by every qubit. Use
`machine.dc_bias.applied_for_qubit("q3")` to apply the selected qubit's voltage
and reliably return channel 0 (physical output 1) to zero afterward.

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
python apps/profile_studio/server.py
```

Then open <http://127.0.0.1:8766>. The HTML editor exposes the profile files as
Profile, Qubits, Pulses, and Connectivity tabs and saves changes back to the
selected profile JSON.


## Two readout modes

Experiments select `parameters.readout_states = ["g", "e"]` (default) or
`["g", "e", "f"]`. The selection chooses the measurement pulse, its IQ centers,
frequency, and active-reset basis together. `reset_type="active"` is sufficient;
`active_gef` is a legacy alias and also follows `readout_states`.

| Setting | G/E | G/E/F |
| --- | --- | --- |
| Operation / pulse in `pulses.json` | `readout` | `readout_GEF` |
| RF frequency in `qubits.json` | `frequencies_hz.resonator` | `readout_gef.frequency_hz` |
| Amplitude / duration | `pulses.<qubit>.readout.amplitude` / `length_ns` | `pulses.<qubit>.readout_GEF.amplitude` / `length_ns` |
| Centers / assignment matrix | `readout.gef_centers` / `confusion_matrix` | `readout_gef.gef_centers` / `confusion_matrix` |
| Angle / depletion / kernel switch | `readout` section | `readout_gef` section |
| Optimized kernel file | `kernels/<qubit>_readout_kernel.npz` | `kernels/<qubit>_readout_GEF_kernel.npz` |

Both centers arrays use demodulation units; their row order is G/E or G/E/F.
The legacy name `readout.gef_centers` is retained for compatibility but belongs
to the GE pulse. It is never used as the new GEF pulse's calibration. GEF uses
nearest-center classification. Time of flight, smearing, ports, LO, and input
gain remain shared resonator/hardware settings.

New GEF pulses start with a copy of the existing amplitude and length, flat
weights, zero integration angle, and the existing resonator frequency plus
its legacy GEF shift. These are initial settings, not a calibrated GEF readout.
The new GEF centers and confusion matrix are deliberately null. Tune frequency,
amplitude, and duration, then acquire all three IQ clouds using thermal reset.
Changing the length also requires integration weights covering the new length;
optimized kernels must be recalibrated for that duration.

Profiles with missing centers can load for calibration. Discrimination and
active reset reject a mode without its calibrated centers. New IQ calibrations
also save `calibration_signature`; a change to frequency, amplitude, duration,
weights, integration angle, acquisition timing, smearing, input gain, or output power rejects the stale
calibration. Older GE calibrations without this signature remain readable.
Readout tuning proposals clear the selected mode's centers/matrix, requiring
fresh IQ calibration before discrimination. Check profile proposals before
applying them; no hardware calibration is launched by changing these files.
