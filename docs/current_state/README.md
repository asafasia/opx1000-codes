# Current State Notes

This folder contains dated lab-state snapshots. Use one dated folder per day or
measurement campaign so the current device status, known blockers, and open
questions are not mixed with durable hardware documentation.

Suggested date format:

```text
DD_MM_YYYY
```

Each dated note should record:

- Which qubits are calibrated well enough to use.
- Which measurements are currently blocked.
- Which calibration values or profile branch were used.
- Any strange observations that need follow-up.
- The next validation experiment to run.

These notes are working lab records. The durable source of truth for executable
parameters remains the profile JSON under `profiles/`.
