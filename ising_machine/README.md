# Classical Analog Ising Machine

This package contains QUA macros for building a classical Ising-machine
experiment without real qubits or calibrated quantum operations.

For the initial one-spin experiment, an analog-like value represents the
binary state:

```text
signal > threshold  -> state 1 -> Ising spin -1
signal <= threshold -> state 0 -> Ising spin +1
```

For example:

```text
signal 1.0 with threshold 0.75 -> state 1
signal 0.5 with threshold 0.75 -> state 0
```

The macros in `utils.py` classify signals, convert binary states into Ising
spins, calculate energy, calculate the cost of a flip, and perform a
zero-temperature update.

They operate on QUA variables and do not depend on the QuAM profile, physical
qubits, or calibrated readout.

Run the first one-spin experiment:

```powershell
python -m ising_machine.one_spin_experiment
```

The experiment alternates synthetic inputs `0.5` and `1.0`, classifies them
using threshold `0.75`, performs a zero-temperature Ising update, emits a pulse
representing the updated state, and plots the saved state, energy history, and
simulated analog waveforms. For every iteration, the waveform contains the
synthetic input pulse followed by the updated-state pulse.

The experiment saves its results to:

```text
data/ising_machine/one_spin_experiment.csv
```

Compare those results against an independent pure-Python reference model:

```powershell
python -m ising_machine.one_spin_reference
```

The comparison checks every state, spin, flip decision, energy, and flip cost,
then plots experiment values beside the expected values. If the experiment CSV
does not exist yet, the reference script runs and plots the pure-Python model
by itself.
