# Readout-Error Mitigation

Readout-error mitigation corrects discriminated qubit populations for known
classification errors. In this repository it is an optional post-processing
step shared by qubit calibrations through the
`use_readout_mitigation` parameter.

This implementation currently supports independent binary `g/e` readout. It
does not mitigate raw `I/Q` data, correlated multi-qubit readout errors, or
three-state `g/e/f` discrimination.

## Assignment-Matrix Convention

Binary IQ-blobs analysis prepares `g` and `e`, discriminates every shot, and
produces the threshold-based fidelity matrix

```text
    measured state
         g     e
g     [ Mgg   Mge ]
e     [ Meg   Mee ]
^ prepared state
```

Each entry is a conditional probability:

```text
Mij = P(measured state j | prepared state i)
```

The rows should therefore sum to approximately one. The matrix is saved in the
profile as:

```text
qubits.json.qubits.<qubit>.readout.confusion_matrix
```

Despite the profile field's historical `confusion_matrix` name, the stored
binary matrix is the `fidelity_matrix` calculated using the same rotated-I
threshold employed by on-the-fly state discrimination. The nearest-IQ-center
matrix used by `g/e/f` analysis is not used for binary mitigation.

## Correction

For a row-vector population convention,

```text
p_measured = p_true @ M
p_true      = p_measured @ inverse(M)
```

The discriminated dataset contains the measured excited-state probability
`state = P(measured e)`. For a row-normalized binary matrix, its corrected value
is equivalently

```text
P(true e) = (P(measured e) - Mge) / (Mee - Mge)
```

The shared calibration lifecycle performs the full 2x2 inverse calculation for
each selected qubit. It replaces `state` with the corrected population before
analysis and plotting and keeps the original values in
`state_unmitigated`.

Corrected values are intentionally not clipped to `[0, 1]`. Finite-shot noise
can produce small excursions beyond the physical interval, and clipping those
values would bias fits and conceal uncertainty or a poor readout calibration.

## Workflow

### 1. Calibrate binary readout

Run IQ blobs for the target qubit using the same readout operation and reset
conditions relevant to the later experiment. This is a real hardware
calibration; review the result and profile proposal before applying it.

```powershell
python -m calibrations.runner run iq-blobs --qubit q9 --set states='["g","e"]'
```

After accepting the proposed profile update, confirm that the qubit's
`readout.confusion_matrix` is present in `profiles/<profile>/qubits.json`.

### 2. Enable discrimination and mitigation

Both switches are required:

```powershell
python -m calibrations.runner run power-rabi --qubit q9 `
  --set use_state_discrimination=true `
  --set use_readout_mitigation=true
```

The equivalent Python configuration is:

```python
parameters.use_state_discrimination = True
parameters.use_readout_mitigation = True
```

The default is `False`, so existing calibration behavior is unchanged.

## Data Lifecycle

During a new acquisition, the unmitigated dataset is saved first as raw data.
Mitigation is then applied in memory before analysis and plotting. The processed
dataset exposes both:

- `state`: mitigated excited-state population.
- `state_unmitigated`: population returned by hardware discrimination.

When a saved run is loaded with `use_readout_mitigation=True`, the same
correction is applied using the matrix from the currently loaded profile. This
allows raw data to remain unchanged and avoids applying mitigation twice.

The mitigated `state` variable carries these attributes:

```text
readout_mitigated = true
readout_mitigation_method = "inverse_assignment_matrix"
```

## Validation And Failure Modes

Mitigation stops with a `CalibrationError` instead of silently returning
uncorrected data when:

- `use_state_discrimination` is `False`;
- the dataset does not contain `state`;
- no matrix is available for a selected qubit;
- the matrix is not finite and 2x2;
- the matrix is singular;
- multiple qubits are selected but the dataset has no `qubit` dimension.

A matrix can be invertible yet still be poorly conditioned. When `Mge` and
`Mee` are close, inversion strongly amplifies shot noise. Treat very large
corrections or unstable values as evidence that readout should be recalibrated,
not as reliable populations.

## Interpretation Limits

The IQ-blobs matrix includes errors from state preparation as well as readout,
especially the pulse used to prepare `e`. Its inverse therefore corrects the
measured assignment behavior of that calibration sequence; it is not a pure,
independent measurement-error model unless preparation error is negligible.

For simultaneous multi-qubit experiments, the current implementation applies
one independent 2x2 matrix per qubit. It does not model correlated assignment
errors. A correlated two-qubit correction would require a calibrated 4x4 joint
assignment matrix and a different data representation.
