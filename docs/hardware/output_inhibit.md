# OPX Output Inhibit During Refrigerator Heating

Use this procedure before heating or opening the dilution refrigerator. The
software latch prevents repository calibrations from starting real OPX jobs
and closes all currently open quantum machines (QMs).

> **Important:** This is a software safeguard, not a hardware interlock. If
> zero RF/DC output must be guaranteed, also disable or disconnect the physical
> RF/DC sources and verify the signal chain according to the lab procedure.

## Before heating the refrigerator: one-click method

From the repository root, double-click:

```text
SAFE_FOR_FRIDGE_HEATING.cmd
```

Do not start heating until the window displays both lines:

```text
SUCCESS: SOFTWARE OUTPUT INHIBIT IS ON
SUCCESS: ALL OPEN QUANTUM MACHINES ARE CLOSED
```

If it displays `SAFETY CHECK FAILED`, do not heat. The software latch is on,
but the script could not verify that existing hardware jobs were stopped.

Leave the inhibit enabled throughout the heating period. The script never
re-enables outputs.

## Terminal method

Open PowerShell in the repository root:

```powershell
cd C:\Users\owner\Developer\opx1000-codes
$env:PYTHONPATH = (Get-Location).Path
```

Engage the inhibit:

```powershell
python -m calibrations.runner outputs inhibit --reason "dilution refrigerator heating"
```

The command performs these actions in this order:

1. Records the local `outputs inhibited` latch.
2. Connects to the QOP cluster selected by the profile.
3. Closes all open QMs, stopping their jobs and returning their programmed
   outputs to the closed-QM state.
4. Asks QOP for the open-QM list and fails unless that list is empty.

For a non-default profile, specify it explicitly:

```powershell
python -m calibrations.runner outputs inhibit --profile main --reason "dilution refrigerator heating"
```

Do not use `outputs enable` as part of the heating procedure.

## Verify the latch

Checking the latch does not contact the hardware:

```powershell
python -m calibrations.runner outputs status
```

Expected output:

```text
outputs: inhibited
engaged_at: ...
reason: dilution refrigerator heating
```

Optionally verify that QOP reports no open QMs or active jobs:

```powershell
python -m calibrations.runner jobs
```

Expected summary:

```text
open_qms: 0
active_jobs: no
jobs: none
```

If the `outputs inhibit` command reports a network or QOP error, the local
latch was still written first, so new repository calibrations are blocked.
However, closure of existing QMs was not verified. Stop the jobs through the
QOP interface or follow the lab's physical shutdown procedure before heating.

## Behavior while inhibited

Real class-based calibrations stop with an `OutputsInhibitedError` before QUA
execution. These activities remain available:

- QUA simulation with `--simulate`.
- Calibration `--dry-run` checks.
- Loading and analyzing already saved data.
- Editing and validating profiles.

The latch does not change `profiles/main` or replace calibrated pulse values
with zeros. This avoids accidentally restoring stale values after cooldown.

## Re-enable after cooldown

First verify independently that:

- The refrigerator is cold and stable.
- The RF/DC wiring and attenuation chain are restored and safe.
- No physical interlock or source disable should remain engaged.
- The intended profile and active qubits are correct.

Then clear the software latch with the required acknowledgement:

```text
ENABLE_OUTPUTS_AFTER_FRIDGE_COLD.cmd
```

Double-click the file and answer `Y`. Alternatively, use PowerShell:

```powershell
python -m calibrations.runner outputs enable --confirm-fridge-cold
```

Confirm the state:

```powershell
python -m calibrations.runner outputs status
```

Expected output:

```text
outputs: enabled
```

Re-enabling only removes the software block. It does not start a calibration,
open a QM, or emit a pulse.

## Quick reference

| Action | Command |
| --- | --- |
| Inhibit and close QMs | `python -m calibrations.runner outputs inhibit --reason "dilution refrigerator heating"` |
| Check latch | `python -m calibrations.runner outputs status` |
| Check QOP jobs | `python -m calibrations.runner jobs` |
| Re-enable after verification | `python -m calibrations.runner outputs enable --confirm-fridge-cold` |
