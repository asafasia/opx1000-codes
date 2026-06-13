# Dynamic Circuit: Separate Active Reset

`active_reset.py` is a Qualibration node organized like
`calibrations/07_iq_blobs_separate.py`.

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
