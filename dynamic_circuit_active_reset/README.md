# Dynamic Circuit: Active Reset

This experiment demonstrates a simple dynamic circuit on `q1`:

1. Prepare `|1>` for half the shots and leave `|0>` unchanged for the rest.
2. Measure the qubit.
3. If the measured state is `1`, apply the configured `x180` pi pulse.
4. Otherwise, do nothing.
5. Measure again to verify that the qubit was reset to `|0>`.

The conditional branch is:

```python
with if_(initial_i > threshold):
    qubit.xy.play("x180")
```

The threshold comes from:

```python
qubit.resonator.operations["readout"].threshold
```

Before running on hardware, make sure this threshold is calibrated and that
state `|1>` corresponds to `I > threshold`. If the readout polarity is
reversed, change both comparisons in `active_reset.py` to `I < threshold`.

Simulate the sequence:

```powershell
python -m dynamic_circuit_active_reset.active_reset --num-shots 10
```

Simulation automatically plots the raw simulated samples. If the QOP cannot
provide raw samples, it falls back to plotting the waveform report. The script
waits for simulation completion before requesting either plot. Each simulated
controller is shown in its own subplot.

Run without opening a plot with:

```powershell
python -m dynamic_circuit_active_reset.active_reset --num-shots 10 --no-plot
```

Execute it on hardware:

```powershell
python -m dynamic_circuit_active_reset.active_reset --execute
```
