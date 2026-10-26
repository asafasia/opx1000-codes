# Or_profile

Or chip profile for the wiring recorded on 25 October 2026. The original
`q1`, `q2`, and `q3` entries are now named **q1b**, **q2b**, and **q3b**.
On 26 October 2026, independent **q1a**, **q2a**, and **q3a** copies were added
with the same parameters, pulses, metrics, and connectivity as their B counterparts.
Use these exact lowercase keys, including the `q` prefix, to select a qubit.

| Profile key | Chip label | Drive output | Readout output | Readout input |
| --- | --- | --- | --- | --- |
| q1b | 1B | 3 | 8 | 2 |
| q2b | 2B | 5 | 8 | 2 |
| q3b | 3B | 7 | 8 | 2 |
| q1a | 1A | 3 | 8 | 2 |
| q2a | 2A | 5 | 8 | 2 |
| q3a | 3A | 7 | 8 | 2 |

All ports are on `con1`, MW-FEM slot 7. Wiring follows the
[dated port assignments](../../docs/hardware/opx1000_ports_2026-10-25.md).
Only Or's chip is included; `q3b` is active by default.
`Or_profile` is the repository default profile. Single-qubit selection
is supported for any of the six entries. A/B counterparts initially share the
same physical ports and display grid positions. The A/B qubits use the same
lines, with connections switched manually. Subsequent calibration updates affect only the selected entry.

## Parameter source

Imported on 25 October 2026 from the sibling repository:

- `../opx1000_Or/profiles/single_qubit/general.json`: network, controller/FEM,
  bands, sampling rate, power, and acquisition gain setting.
- `../opx1000_Or/profiles/single_qubit/pulses.json`: each qubit's frequencies,
  LOs, timing, square readout, saturation, and square pi pulse.
- `../opx1000_Or/machine/machine.py`: square pulse shapes, zero smearing,
  and unrotated constant integration weights.

The source's common drive port 7 is replaced by the per-qubit assignments
above. Source operation `pi` maps to `x180` using `x180_const`.
Drive and readout full-scale power are both 10 dBm. Readout uses band 2
and a shared 6.3 GHz LO; drive uses band 1 with per-output LOs.
Input gain remains unspecified, matching the source.

Original imported values for the entry now named `q3b` (see JSON for current values):

| Parameter | Value |
| --- | --- |
| Qubit frequency | 3,904,451,423 Hz |
| Drive LO | 4,000,000,000 Hz |
| Readout frequency | 6,419,800,000 Hz |
| Readout LO | 6,300,000,000 Hz |
| Square pi pulse | Amplitude 0.99, length 400 ns |
| Square readout pulse | Amplitude 0.06, length 2,000 ns |
| Saturation pulse | Amplitude 0.1, length 100,000 ns |
| Time of flight | 320 ns |
| Readout depletion | 2,500 ns |
| Thermalization | 300,000 ns |

The source does not provide calibrated discrimination, coherence,
anharmonicity, or G/E/F settings. Unknown values remain null; required
threshold fields are zero placeholders. Nearest-center discrimination is
selected, with centers unset, so acquire IQ blobs before using state
discrimination or active reset. G/E/F and EF operations are not defined.
Grid locations are display-only positions, not measured chip coordinates.

## Use and validation

```python
from quam_config import CreateMachine
from quam_config.create_machine_from_profile import create_machine_from_profile

cm = CreateMachine("Or_profile")  # All six entries, q3b active
machine = create_machine_from_profile("Or_profile", qubit="q2b", save=False)
machine_a = create_machine_from_profile("Or_profile", qubit="q2a", save=False)
```

`CreateMachine()` uses this profile by default. `CreateMachine(qubit="q3b")`
or `CreateMachine("Or_profile", qubit="q3b")` builds only Or 3B. Explicit
profile choices and `QUAM_PROFILE` environment overrides remain supported.

```powershell
python -m profiles.validate_profile Or_profile
python -m profiles.validate_profile Or_profile --qubit q3b
python -m quam_config.create_machine_from_profile --profile Or_profile --no-save
python -m quam_config.create_machine_from_profile --profile Or_profile --qubit q3b --no-save
```

Profile creation and validation do not run hardware calibrations.
