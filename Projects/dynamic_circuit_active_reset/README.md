# Dynamic Circuit: Separate Active Reset

`active_reset.py` is a Qualibration node organized like
`calibrations/07_iq_blobs_separate.py`.

Active reset prepares the qubit in `|g>` using measurement and feedback instead
of waiting for passive thermal relaxation. The program measures the qubit, uses
the configured readout threshold to decide whether the qubit is in `|e>`, and
plays `x180` only when the measured state is excited.

It runs two independent jobs:

1. Ground-prepared acquisition.
2. Excited-prepared acquisition.

Each shot:

1. Measures and saves the initial IQ point and thresholded state.
2. Applies `x180` only when the initial I value is above the configured
   readout threshold.
3. Measures and saves the final IQ point and thresholded state.

The saved dataset contains before/after IQ clouds, measured states, and whether
the reset pulse was applied for both preparations. The generated figure compares
the IQ clouds and measured excited-state fractions before and after reset.

The active-reset decision uses:

```python
with if_(initial_i[i] > threshold):
    qubit.xy.play("x180")
```

The threshold and integration-weight rotation come from the selected device
profile. Calibrate them with the IQ-blobs experiment before evaluating reset
performance.

## What The Figure Means

The output figure compares each qubit before and after the conditional reset:

- The "Before active reset" panel shows the state distribution immediately
  after the initial preparation and first measurement.
- The "After active reset" panel shows the distribution after the conditional
  `x180` feedback and second measurement.
- A good result has much less excited-state population after reset, especially
  in the excited-prepared acquisition.
- If the ground-prepared data gets worse after reset, the threshold, IQ
  rotation, or `x180` pulse is probably not reliable enough yet.

## Prerequisites

Run this validator only after the basic calibrations are usable:

- Resonator spectroscopy has selected a readout frequency.
- Qubit spectroscopy has selected a qubit drive frequency.
- Rabi or pi-train calibration has produced a reasonable `x180`.
- IQ blobs have produced a threshold and integration-weight angle.
- The selected profile contains the qubit or qubits you want to validate.

## How To Use The Result

Use this node as a validation step before enabling active reset broadly in other
calibrations.

1. Run normal calibrations with `reset_type = "thermal"` during early bring-up.
2. Run IQ blobs and confirm the blobs are separated.
3. Run `Projects/dynamic_circuit_active_reset/active_reset.py`.
4. Check that the excited-prepared shots move mostly to ground after reset.
5. If the result is good, use `node.parameters.reset_type = "active"` in later
   calibrations.
6. If the result is bad, keep `reset_type = "thermal"` and recalibrate readout
   threshold, integration weights, and `x180`.

This node does not update profile values by itself. It is a diagnostic and
validation experiment.
