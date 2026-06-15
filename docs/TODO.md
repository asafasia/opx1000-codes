# Documentation TODO

This file tracks documentation and repository-organization follow-ups. Keep it
practical: add items that would help the next person run, debug, or maintain
the calibration workflow.

## Calibration Docs

- [ ] Add a per-script calibration reference table:
  script name, purpose, required previous calibration, profile fields read,
  profile fields updated, expected outputs, and what a good result looks like.
- [ ] Add a bring-up checklist for a new cooldown or new branch:
  resonator spectroscopy, qubit spectroscopy, Rabi, T1, Ramsey, IQ blobs,
  readout optimization, DRAG, and randomized benchmarking.
- [ ] Add troubleshooting notes for common calibration failures:
  missing qubit resonance, weak resonator contrast, bad Rabi fit, empty IQ
  blobs, failed active reset, and acquisition/readout synchronization issues.
- [ ] Add a short note explaining when to use broad/high-amplitude scans versus
  narrow/low-amplitude fine-tuning scans.

## Profile Docs

- [ ] Add a field-by-field profile reference for:
  `profile.json`, `connectivity.json`, `qubits.json`, and `pulses.json`.
- [ ] Document exactly which calibration updates should modify `qubits.json`
  versus `pulses.json`.
- [ ] Document the profile update flow:
  calibration result -> proposal -> review -> apply -> validate -> commit.
- [ ] Add examples for editing profile values with Profile Studio.
- [ ] Add examples for editing profile values programmatically with
  `Profile.load()`, `Profile.save()`, and `ProfileUpdater`.

## Collaboration

- [ ] Write a branch workflow guide for multiple users calibrating in parallel.
- [ ] Define when a user's branch profile values are ready to merge into the
  trusted profile.
- [ ] Add a checklist for reviewing profile changes before merge.
- [ ] Add a convention for recording calibration run IDs in current-state notes.

## Data And Tools

- [ ] Document the saved calibration data layout under `data/calibrations/`.
- [ ] Document calibration update proposals under `data/calibration_updates/`.
- [ ] Add a quick-start guide for `apps/visualiser`.
- [ ] Add a quick-start guide for `apps/profile_studio`.
- [ ] Explain what is generated output versus durable source-of-truth input.

## Repository Structure

- [ ] Finish documenting the new structure:
  `apps/`, `calibration_io/`, `Projects/`, `profiles/`, and `utils/`.
- [ ] Decide whether old generated folders such as root `visualiser/` logs and
  `updater/__pycache__/` should be cleaned locally.
- [ ] Consider moving `calibration_io/test_calibration_saver.py` into `tests/`
  if all tests should live in one place.
- [ ] Consider adding a short architecture diagram for the flow:
  profiles -> CreateMachine -> calibration script -> data save -> profile
  update proposal.

## Current Known Issues To Document

- [ ] Add deeper notes on the acquisition/readout-pulse synchronization bug.
- [ ] Record validation results for active reset on `q1`, `q2`, and `q9`.
- [ ] Record investigation notes for qubits that show resonator separation but
  no clear qubit spectroscopy resonance.
