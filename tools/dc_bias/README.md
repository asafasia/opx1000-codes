# Arduino DC-bias control

The reusable driver in `hardware/arduino_dc_bias.py` sends commands in the
existing Arduino format:

```text
SET,<channel>,<voltage>\r
```

`ArduinoDCBias` in `quam_config/components/dc_bias.py` exposes the controller as
a host-side QuAM component. Its live serial connection is not serialized and it
does not add anything to the QUA configuration.

The controller supports channels 0 through 7 and enforces a hard maximum
magnitude of 0.01 V. The selected DC-bias output is hardcoded as channel 0 for
all qubits. When an `applied_for_qubit(...)` block ends—or is interrupted—the
component returns channel 0 to zero and closes the connection it opened.

## Setup

Install the serial dependency in the active lab Python environment:

```powershell
python -m pip install -r tools/dc_bias/requirements.txt
```

## One-minute example

The serial port and voltage limit come from `connectivity.dc_bias`. Each
single-qubit entry supplies only its voltage through `dc_bias_v`. Edit the
constants near the top of `dc_bias.py` to select the qubit and hold time:

```python
QUBIT = "q3"
HOLD_TIME_S = 60.0
```

Set the requested voltage in `profiles/single_qubit/qubits.json`:

```json
"q3": {
  "dc_bias_v": 0.005
}
```

Run it from the repository root without command-line arguments:

```powershell
python tools/dc_bias/dc_bias.py
```

Use decimal volts: `0.005` means 5 mV. The utility rejects requested voltages
outside ±0.01 V before opening the serial port. Verify that the configured
limit and requested voltage are safe for the connected hardware before running.

Calibration or experiment code can use the same component:

```python
machine = create_machine(qubit="q3")

with machine.dc_bias.applied_for_qubit("q3"):
    run_experiment()
```

Class-based calibrations do not need this explicit context. `BaseCalibration`
automatically applies a nonzero profile bias around real hardware execution and
zeros it afterward. Simulation, dry-run, loaded-data analysis, and zero profile
values leave the serial output untouched.

The serial bias is host-controlled and is not synchronized to QUA timing. Use
an OPX/LF-FEM analog output instead when a bias must change during a QUA program.
