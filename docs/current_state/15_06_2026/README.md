# Current State: 15_06_2026

This note records the device and calibration state on 15 June 2026. It is a
working lab snapshot, not a replacement for the profile JSON files.

## Main Blocker

There is still a strange synchronization bug between acquisition and the readout
pulse. The suspected issue is that the acquire window and the readout pulse are
not aligned or synchronized correctly.

This currently prevents reliable measurement of:

- T1
- T2 / Ramsey-style coherence measurements
- Gate fidelity
- Any later calibration that depends on trusted time-domain readout

Until this is fixed, downstream calibration results should be treated with care,
even when earlier spectroscopy and Rabi-style results look reasonable.

## Calibrated Qubits

Qubits `q1`, `q2`, and `q9` are calibrated well enough to use as the current
working set.

Active reset is an important validator for the total calibration quality on
these qubits. A good active-reset result checks that several pieces work
together:

- Readout frequency and amplitude are usable.
- IQ separation is meaningful.
- The readout threshold is sensible.
- The qubit drive pulse can return the qubit to the target state.
- Timing and depletion are reasonable enough for repeated shots.

Because active reset depends on both measurement and control, it should be used
as a practical end-to-end validation after the basic bring-up calibrations.

## Other Qubits

For the other qubits, the qubit resonance has not been found yet.

Some of these qubits do show resonator-response separation in resonator
spectroscopy, which is strange because it suggests that the resonator can see a
state-dependent response, but the direct qubit spectroscopy resonance is still
missing or unclear.

This may point to one or more of:

- Incorrect qubit frequency search range.
- Spectroscopy amplitude or pulse length not suitable for the first-pass scan.
- Wrong or stale profile values for LO, IF, port, or selected qubit.
- Readout threshold or integration weights giving misleading contrast.
- Timing/synchronization issues affecting the apparent spectroscopy signal.

## Immediate Next Steps

1. Investigate and fix the acquisition/readout-pulse synchronization bug.
2. Revalidate `q1`, `q2`, and `q9` with active reset after the timing fix.
3. For the remaining qubits, repeat broad qubit spectroscopy from a trusted
   resonator/readout point.
4. Compare qubits that show resonator separation but no qubit resonance against
   their profile entries: frequency range, LO, IF, ports, readout pulse, and
   active-qubit selection.
5. Record the profile branch and exact calibration run IDs when updating this
   dated state.
