# Echo-Lorentzian QuTiP Simulation

This folder is a PC-only simulator for the echo-Lorentzian sweep. It uses the
same main parameter names as `Projects/echo-lorentizan`, but it does not connect
to the OPX and does not run a real experiment.

The simulator solves a driven two-level Hamiltonian with QuTiP:

```text
H / hbar = pi * detuning * sigma_z + pi * Omega(t) * sigma_x
```

where `Omega(t)` is calibrated from the square `x180` pulse:

```text
Omega_hz(t) = waveform_voltage(t) / x180_amplitude / (2 * x180_length)
```

Run:

```powershell
python -m pip install -r Projects\echo-lorentzian-qutip-simulation\requirements.txt
python Projects\echo-lorentzian-qutip-simulation\simulate_echo_lorentzian.py --pulse-shape gaussian --echo --lorentzian-length-in-ns 160000 --waveform-template-length-in-ns 60000 --lorentzian-peak-amplitude 0.5 --amp-factor-step 0.01 --frequency-span-in-mhz 10 --frequency-step-in-mhz 0.2
```

Outputs are written under:

```text
Projects/echo-lorentzian-qutip-simulation/output/
```

The output includes:

- `parameters.json`
- `echo_lorentzian_qutip.nc`
- `echo_lorentzian_qutip.png`

For quick local testing, reduce the sweep size and pulse length:

```powershell
python Projects\echo-lorentzian-qutip-simulation\simulate_echo_lorentzian.py --pulse-shape root_lorentzian --echo --lorentzian-length-in-ns 80 --cutoff 0.25 --amp-factor-step 0.25 --frequency-span-in-mhz 20 --frequency-step-in-mhz 5
```
