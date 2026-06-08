# Ising Simulation

A compact, readable Monte Carlo simulation of the classical two-dimensional
Ising model. It uses the Metropolis algorithm on a square lattice with periodic
boundaries.

The model energy is:

```text
E = -J sum_<i,j> s_i s_j - h sum_i s_i,   s_i in {-1, +1}
```

For `J = 1` and `h = 0`, the infinite 2D model has a critical temperature near
`T_c = 2.269`. A finite simulation does not show a perfectly sharp transition,
but a temperature sweep makes the change easy to see.

## Run

From the repository root:

```bash
python3 -m ising_simulation.run_sweep
```

Direct execution also works from any directory:

```bash
python3 /path/to/opx1000-codes/ising_simulation/run_sweep.py
```

The default sweep concentrates samples around the critical region and writes:

```text
data/ising_simulation/temperature_sweep.csv
data/ising_simulation/temperature_sweep.png
```

Dependencies are NumPy and Matplotlib. The default run uses a `24 x 24` lattice
and is intentionally moderate. For a quick experiment, reduce
`equilibration_sweeps` and `sample_sweeps` in `run_sweep.py`. For cleaner
physics, increase the lattice size and sweep counts.

```bash
python3 -m pip install -r ising_simulation/requirements.txt
```

Run the small correctness checks with:

```bash
python3 -m ising_simulation.test_model
```

## Interesting Observables

- **Mean energy per spin** falls as neighboring spins align at low temperature.
- **Absolute magnetization per spin** is close to one in an ordered state and
  approaches zero in a disordered state. Absolute magnetization avoids
  cancellation when a finite lattice switches between positive and negative
  ordered states.
- **Energy variance** measures fluctuations in energy.
- **Heat capacity** is energy variance divided by `N T^2`; it peaks near the
  transition.
- **Magnetization variance** measures fluctuations in total magnetization.
- **Susceptibility** is magnetization variance divided by `N T`; it grows near
  the transition.
- **Nearest-neighbor correlation** measures local alignment.
- **Correlation length** estimates how far spin correlations extend. Its simple
  exponential-fit estimator is useful for exploration, but is noisy on small
  lattices.
- **Acceptance rate** shows how often Metropolis proposals are accepted. Low
  temperatures strongly reject energy-increasing flips.

All reported thermodynamic quantities use units where Boltzmann's constant
`k_B = 1`.

## Files

- `model.py`: lattice, energy calculation, and Metropolis solver.
- `observables.py`: thermodynamic metrics and spatial correlations.
- `sweep.py`: reusable temperature sweep, CSV output, and plotting.
- `run_sweep.py`: practical default experiment.
- `test_model.py`: lightweight correctness checks.
