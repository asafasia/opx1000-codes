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

#### Broad-To-Narrow Qubit Spectroscopy Workflow

Use `calibrations/03a_qubit_spectroscopy.py` as a search tool first, then as a
fine calibration. The first goal is not a perfect linewidth; it is to see a
real resonance somewhere in the scan.

The main parameters to change are:

- `frequency_span_in_mhz`: total scan width around the current profile qubit
  frequency.
- `frequency_step_in_mhz`: spacing between scan points.
- `operation_amplitude_factor`: linear multiplier on the configured
  spectroscopy operation amplitude.
- `operation_len_in_ns`: optional pulse length override. Leave it as `None`
  unless you deliberately want to change the saturation-pulse duration.
- `num_shots`: averaging. Increase this when the resonance is visible but noisy.

For an unknown or poorly trusted qubit frequency, start broad and strong:

- Set `frequency_span_in_mhz` to a broad range, typically `100` to `500`.
- Use a large `operation_amplitude_factor`, often around `1.0`.
- Use a coarse `frequency_step_in_mhz`, for example `1` to `5`, so the scan
  finishes quickly.
- Keep `operation_len_in_ns = None` at first, so the script uses the configured
  spectroscopy pulse length.

This high-power scan intentionally power-broadens the qubit. The peak can be
wide and the fitted FWHM can be ugly; that is acceptable at this stage. What
matters is that the peak position is physically reasonable and repeatable.

Once the resonance is found, move toward a cleaner spectrum gradually. Reduce
the spectroscopy power linearly, not in one jump. For example:

| Pass | `frequency_span_in_mhz` | `frequency_step_in_mhz` | `operation_amplitude_factor` | Purpose |
| --- | ---: | ---: | ---: | --- |
| 1 | `500` | `2` to `5` | `1.0` | Find a missing resonance. |
| 2 | `200` | `1` to `2` | `0.7` | Confirm the peak near the new center. |
| 3 | `100` | `0.5` to `1` | `0.4` | Reduce power broadening. |
| 4 | `50` | `0.25` to `0.5` | `0.2` | Improve frequency precision. |
| 5 | `10` to `20` | `0.05` to `0.25` | `0.05` to `0.1` | Final low-power check. |

The exact numbers depend on the qubit and the readout signal, but the shape of
the workflow should stay the same: large span and high power to find the
feature, then smaller span and lower power to measure it accurately.

After each pass:

1. Check that the peak is not sitting on the edge of the scan.
2. If the peak is near the edge, recenter the profile qubit frequency or expand
   the span and repeat.
3. If the peak is centered and repeatable, update the qubit frequency from the
   fitted result.
4. Lower `operation_amplitude_factor` by a linear step and reduce
   `frequency_span_in_mhz`.
5. Reduce `frequency_step_in_mhz` only after the scan is already centered.

Do not trust a final frequency from the first high-power broad scan alone. High
power is useful because it makes the resonance easy to find, but it can shift or
broaden the apparent peak. Use the lower-power passes to decide the value that
should be kept in the profile.

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

## Active Reset

Active reset is a fast way to start each shot with the qubit in the ground
state without waiting for many T1 times. A thermal reset waits long enough for
the qubit to relax naturally. Active reset measures the qubit first and then
uses feedback: if the measurement says the qubit is excited, the program applies
an `x180` pulse to bring it back to ground.

In this repository, the active-reset decision is based on the rotated readout
quadrature and the threshold stored in the selected profile. Conceptually, the
logic is:

```python
measure_readout()
if measured_I > readout_threshold:
    play_x180()
```

This means active reset depends on both readout and control. It should be used
only after these pieces are already reasonable:

- Resonator frequency and readout pulse are calibrated.
- IQ blobs give a usable ground/excited separation.
- The readout threshold and integration-weight angle in the profile are valid.
- The `x180` pulse amplitude and frequency are good enough to flip the qubit.
- The readout delay and resonator depletion time are not obviously wrong.

The main advantage is speed. Once active reset is trustworthy, calibration
experiments can run many shots without inserting a long thermalization wait at
the end of every shot. This is especially useful for Rabi, T1, Ramsey, IQ blobs,
readout optimization, DRAG, and randomized benchmarking.

The main risk is feedback based on bad information. If the threshold is wrong,
the integration weights are poorly rotated, or the `x180` pulse is not
calibrated, active reset can prepare the wrong state and make later calibration
results look confusing. When in doubt, use thermal reset until the readout and
pi pulse are trusted.

### How To Use Active Reset In Calibrations

Many calibration scripts expose `node.parameters.reset_type`. Use:

```python
node.parameters.reset_type = "active"
```

to use active reset, or:

```python
node.parameters.reset_type = "thermal"
```

to use passive thermalization. Set this in the script's `custom_param` section
for local debugging, or through the Qualibrate parameter UI when running from
the GUI.

The calibration programs call:

```python
qubit.reset(
    node.parameters.reset_type,
    node.parameters.simulate,
    # log_callable=node.log,
)
```

at the beginning of the shot or sweep point. With `reset_type = "thermal"`, this
performs the configured thermal wait. With `reset_type = "active"`, it performs
the measurement-based reset sequence provided by the QuAM qubit object.

Use this practical sequence:

1. Start with `reset_type = "thermal"` during early bring-up.
2. Calibrate resonator spectroscopy, qubit spectroscopy, Rabi, and IQ blobs.
3. Check that the IQ-blob threshold and integration-weight angle are sensible.
4. Run the standalone active-reset validator:
   `Projects/dynamic_circuit_active_reset/active_reset.py`.
5. If the before/after plots show that excited population is reduced reliably,
   switch routine calibrations to `reset_type = "active"`.
6. If later data becomes inconsistent, repeat IQ blobs and the active-reset
   validator before trusting downstream calibrations.

Do not treat active reset as a replacement for IQ-blob calibration. Active reset
uses the threshold and rotation from IQ blobs; it does not discover them by
itself.

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
