# Gate Level

Gate-level calibration and experiment utilities live here.

## Minimal Qiskit Circuit

Install the provider in `opx1000_env` first:

```powershell
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' -m pip install qiskit-qm-provider
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' -m pip install qiskit-experiments
```

Check the circuit/backend setup without submitting to the QPU:

```powershell
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' -c "from gate_level.runner import Runner; print(Runner(qubit='q3').backend.target.operation_names)"
```

Run the example circuit:

```powershell
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' gate_level\example_single_qubit.py
```

Run the pi-train example. It builds one circuit for each value in
`PI_COUNTS`, where each circuit contains that many `x` gates, then plots the
excited-state population:

```powershell
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' gate_level\example_pi_train.py
```

Run the standard randomized benchmarking example from Qiskit Experiments:

```powershell
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' gate_level\example_standard_rb.py
```

Use the runner from Python:

```python
from qiskit import QuantumCircuit
from gate_level.runner import Runner, result_counts

circuit = QuantumCircuit(1, 1)
circuit.x(0)
circuit.measure(0, 0)

runner = Runner(qubit="q3")
result = runner.submit(circuit, shots=1000)
print(result_counts(result))
```

For multi-circuit experiments, use `result_counts_list(result)`.

The installed `qiskit-qm-provider` currently uses `iqcc_calibration_tools` inside
its `add_basic_macros()` helper. If that package is not installed, the example
falls back to local single-qubit macros for `x`, `sx`, `rz`, `measure`, `reset`,
and related pulse gates using the pulses in your loaded QuAM profile.
