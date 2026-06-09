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

## Multi-Spin Probabilistic Experiment

`pseudo_spin_experiment.py` implements a classical probabilistic Ising machine
on the controller. It time-multiplexes an arbitrary number of pseudo-spins on
one resonator output:

```text
spin -1 (down) -> readout pulse amplitude scale 0.5
spin +1 (up)   -> readout pulse amplitude scale 1.0
```

The pulses encode classical state only; no physical qubit is prepared or
measured. After emitting one pulse per spin, the QUA program performs one
random-site Metropolis update attempt per pseudo-spin. Randomizing the update
sites avoids directional domain drift caused by always sweeping from the first
spin to the last. A proposed flip with energy change `dE <= 0` is accepted,
while an energy-increasing flip is accepted with probability
`exp(-dE / temperature)`.

Run a 100-spin ring for 100 iterations:

```powershell
python -m ising_machine.pseudo_spin_experiment
```

The default effective temperature is `0.7`.

Show the spin evolution while the OPX job is running:

```powershell
python -m ising_machine.pseudo_spin_experiment --live-plot
```

Use `--live-refresh 0.5` to change the live-plot refresh interval in seconds.
The full observables plot is still shown after the hardware run completes
unless `--no-plot` is supplied.

Run with a linear annealing schedule:

```powershell
python -m ising_machine.pseudo_spin_experiment --temperature 2.0 --final-temperature 0.3
```

The experiment runs on connected OPX hardware by default. Every updated
configuration remains available during the run. By default, analysis and CSV
output include only temperature and magnetization per spin, which keeps
post-processing fast. In this default mode, the OPX calculates magnetization
and streams only temperature and magnetization; complete configurations are
not transferred to the host.

Enable energy, heat capacity, susceptibility, correlation length, the detailed
per-spin CSV, and the complete results plot with:

```powershell
python -m ising_machine.pseudo_spin_experiment --full-analysis
```

`model.py` contains the hardware-independent Ising problem and result classes.
It supports arbitrary symmetric coupling matrices and external fields, making
it the replacement boundary for future real-qubit feedback.

Uniform nearest-neighbor rings use an OPX memory-optimized update that stores
only the spin array, coupling, and field. General coupling matrices remain
supported but require `N x N` controller memory. Shortening the readout pulse
reduces experiment runtime, but does not reduce this coupling-memory usage.

## IQ Amplitude Sanity Check

Run a separate real-hardware acquisition that measures paired readout pulses
with amplitude scales `1.0` and `0.5`:

```powershell
python -m ising_machine.iq_amplitude_sanity_check
```

The experiment acquires integrated I/Q values for both amplitudes, saves the
raw paired shots to `data/ising_machine/iq_amplitude_check.csv`, and plots both
responses in the I/Q plane with their centroids. The printed centroid-magnitude
ratio provides a quick check of the expected amplitude scaling.

## 2D Grid Experiment

`pseudo_spin_2d_experiment.py` implements a separate square/rectangular 2D
nearest-neighbor Ising model with periodic boundaries. Spins are flattened
row-by-row for controller storage, but every update couples to the physical
grid's left, right, up, and down neighbors.

Run the default `32 x 32` grid on real hardware:

```powershell
python -m ising_machine.pseudo_spin_2d_experiment
```

Save every grid configuration so the final 2D domains can be plotted:

```powershell
python -m ising_machine.pseudo_spin_2d_experiment --save-configurations
```

Show the evolving 2D grid while the OPX job runs:

```powershell
python -m ising_machine.pseudo_spin_2d_experiment --live-plot
```

Live plotting automatically enables configuration streaming. Use
`--live-refresh 0.5` to change the refresh interval in seconds. The experiment
also waits `50 ms` between iterations by default so live evolution is easier
to see. Change or disable the controller-side delay with:

```powershell
python -m ising_machine.pseudo_spin_2d_experiment --live-plot --iteration-wait-ms 100
python -m ising_machine.pseudo_spin_2d_experiment --iteration-wait-ms 0
```

Live mode retains only the latest full grid on the result server, rather than
all historical grids. This greatly reduces result memory for large grids.
Use `--save-configurations` without `--live-plot` only when every historical
grid is actually needed.

The 2D experiment emits amplitude-coded readout waveforms with `play()` rather
than `measure()`, because the pseudo-spins do not require acquisition. This
avoids activating the measurement path thousands of times per iteration.

Compared with the 1D ring, the 2D model introduces:

- Four neighbors per spin instead of two, increasing each update's arithmetic.
- Row/column topology and periodic wrapping in two directions.
- Extra row/column and periodic-boundary arithmetic on every update. Neighbor
  indices are calculated on the OPX instead of stored in four large arrays,
  keeping persistent array memory close to the single spin array.
- A qualitatively richer phase transition and 2D domain-wall geometry.
- More expensive visualization and configuration transfer because each saved
  state is a complete grid.

The 2D experiment therefore defaults to lightweight temperature and
magnetization streaming. Complete configurations are optional.
