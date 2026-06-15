# Calibration Routine Hierarchy

This note gives the general order of the important calibration routines. The
details of each experiment live in the calibration scripts and analysis modules,
but the high-level workflow is:

1. General bring-up calibration.
2. Fine tuning calibration.
3. Periodic checks and recalibration after hardware, cooldown, or profile
   changes.

The first bring-up pass should find robust approximate values. Fine tuning then
narrows the sweeps, reduces drive amplitudes where appropriate, and improves the
resolution of the fitted parameters.

## 1. General Bring-Up

General bring-up establishes a usable starting point for the device. The goal is
not to find the final best value for every parameter; it is to find values that
make the next calibration step reliable.

### Resonator Spectroscopy

Start with resonator spectroscopy to find the approximate readout resonator
frequency.

Run the resonator scan both without and with a qubit saturation pulse:

- Without saturation, the scan gives the baseline resonator response.
- With saturation, the qubit population is changed, which helps reveal where
  the resonator response is most distinguishable between qubit states.

Use these scans to choose:

- The approximate resonator frequency.
- A readout frequency near the point of maximum state distinguishability.
- A sensible readout amplitude for the next measurements.

Relevant scripts include:

- `calibrations/02a_resonator_spectroscopy.py`
- `calibrations/02b_resonator_spectroscopy_vs_power.py`

### Qubit Spectroscopy

After the resonator frequency is usable, run qubit spectroscopy to find the
qubit resonance.

For the first pass, use a relatively high spectroscopy amplitude. A stronger
drive broadens the response and makes the resonance easier to see when the
frequency is still uncertain.

After the resonance is visible, reduce the spectroscopy amplitude and narrow the
frequency span. This gives better spectral resolution and a more accurate qubit
frequency.

Relevant script:

- `calibrations/03a_qubit_spectroscopy.py`

### Rabi Calibration

Once the qubit frequency is known, run a Rabi experiment to calibrate the pulse
amplitude for a pi rotation.

Use the first Rabi pass to find the approximate pi-pulse amplitude. After that,
repeat with a narrower amplitude range or a more targeted sweep to fine tune the
`x180` amplitude.

Relevant scripts include:

- `calibrations/04b_power_rabi.py`
- `calibrations/04c_pi_train.py`
- `calibrations/04d_power_rabi_chevron.py`

At this point the first bring-up loop is calibrated enough to support more
specific checks such as T1, Ramsey, IQ blobs, readout optimization, DRAG, and
randomized benchmarking.

## 2. Fine Tuning

Fine tuning starts from the bring-up values and improves precision. The usual
pattern is:

1. Center the sweep around the current best value.
2. Reduce the sweep span.
3. Reduce the drive amplitude when the first-pass value was intentionally high.
4. Increase point density or averaging if the measurement is noisy.
5. Update the profile only after the result is stable and physically sensible.

Typical fine-tuning examples:

- Repeat resonator spectroscopy around the selected readout frequency and check
  that state distinguishability remains high.
- Repeat qubit spectroscopy with lower amplitude and narrower frequency span.
- Repeat Rabi or pi-train calibration around the current pi amplitude.
- Run T1 and Ramsey after a valid pi pulse exists.
- Run IQ blobs and readout optimization after readout and qubit-control
  parameters are reliable.
- Run DRAG and randomized benchmarking after the basic single-qubit gates are
  already working.

## Practical Rule

Use broad, strong, forgiving scans when the parameter is unknown. Once the
feature is found, switch to narrow, lower-amplitude, higher-resolution scans to
turn the approximate value into a calibrated value.

## Good Habits When Calibrating

Calibration values are shared state. Frequencies, amplitudes, thresholds,
integration weights, pulse lengths, and timing values can all affect later
experiments. If several users edit the same profile, one user can accidentally
overwrite parameters that another user already calibrated.

For now, use separate git branches when different users are calibrating or
testing different parameter sets. Treat the profile parameters on each branch as
that user's current working calibration. Merge profile changes only after the
values are reviewed and the intended source of truth is clear.

Recommended habits:

1. Start from the currently trusted branch or profile before changing values.
2. Change only the parameter that the latest calibration actually supports.
3. Keep notes about which run produced a new value.
4. Validate the profile after editing JSON.
5. Avoid mixing bring-up values and fine-tuned values from different users in
   the same branch.
6. Promote a branch's profile values only when the calibration sequence is
   internally consistent.

The safest mental model is that a branch is a calibration workspace. The
profile files on that branch describe one coherent machine state.

## Profiles And Editable JSON

The calibration scripts build the QuAM machine from the selected profile. A
profile is a directory under `profiles/` with four JSON files:

- `profile.json`: profile metadata and the list of active qubits.
- `connectivity.json`: controller, FEM, port, line, network, and LO wiring.
- `qubits.json`: qubit, resonator, readout, frequency, coherence, and threshold
  parameters.
- `pulses.json`: pulse definitions, operation names, amplitudes, lengths, and
  integration weights.

Most calibration updates should land in `qubits.json` or `pulses.json`.
Connectivity should change only when the physical wiring, ports, LOs, or
controller setup changes. The active-qubit list belongs in `profile.json`.

Use explicit units in field names, such as `frequency_hz`, `length_ns`, and
`axis_angle_rad`. This keeps profile edits readable and reduces mistakes when
values move between scripts, notes, and JSON.

Validate after manual edits:

```powershell
python -m profiles.validate_profile main
```

For single-qubit work:

```powershell
python -m profiles.validate_profile single_qubit --qubit q3
```

## Profile Studio

Profile Studio is a local HTML editor for the profile JSON files. It is useful
when changing parameters by hand is awkward or error-prone.

Run it from the repository root:

```powershell
python apps/profile_studio/server.py
```

Then open <http://127.0.0.1:8766>.

The editor shows the four profile files in structured tabs: Profile, Qubits,
Pulses, and Connectivity. It edits only existing complete profiles under
`profiles/`. Saves are atomic, must contain valid JSON, and are rejected if the
file changed on disk after it was loaded. This helps prevent silent overwrites
when several people are working.
