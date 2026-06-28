# Calibrations v2

`calibrations_v2` is the class-based replacement for the repeated
`@node.run_action` scripts in `calibrations/`.

Each new calibration should subclass `BaseCalibration` and keep the experiment
logic in ordinary methods:

```python
from calibrations_v2 import BaseCalibration


class ResonatorSpectroscopy(BaseCalibration):
    def create_qua_program(self):
        qubits = self.get_qubits()
        self.namespace["sweep_axes"] = {...}
        with program() as qua_program:
            ...
        return qua_program

    def analyse_data(self):
        self.results["ds_raw"] = process_raw_dataset(self.results["ds_raw"], self)
        self.results["ds_fit"], self.results["fit_results"] = fit_raw_data(
            self.results["ds_raw"],
            self,
        )

    def profile_updates(self):
        return {
            f"qubits.json.qubits.{q.name}.frequencies_hz.resonator": float(...)
            for q in self.namespace["qubits"]
            if self.outcomes.get(q.name) == "successful"
        }
```

The base owns the common lifecycle:

1. optional local parameter edits
2. create QUA program
3. simulate or execute
4. save raw xarray results with a profile snapshot
5. load saved runs
6. analyse data
7. save figures
8. stage and optionally apply profile updates
9. cleanup temporary machine changes

The instance intentionally exposes `parameters`, `machine`, `namespace`,
`results`, `outcomes`, `log()`, and `record_state_updates()` so existing
analysis utilities can migrate gradually.

Useful inherited helpers include `get_qubits()`, `execute_qua_program()`,
`simulate_qua_program()`, `save_raw_results()`, `save_arrays()`,
`save_figures()`, `save_qua_debug_script()`, and `propose_profile_update()`.

Runtime behavior can be controlled with `CalibrationOptions`:

```python
from calibrations_v2 import CalibrationOptions

options = CalibrationOptions(
    save_raw_data=False,
    save_figures=False,
    plot_data=False,
    update_state=False,
    propose_profile_update=False,
)
```

Pass `options=options` into any v2 calibration constructor.

## Terminal runner

The lightweight terminal wrapper is meant for Codex and quick lab use:

```powershell
python -m calibrations_v2.runner list
python -m calibrations_v2.runner describe resonator
python -m calibrations_v2.runner run resonator --qubit q9 --set num_shots=200
python -m calibrations_v2.runner run power-rabi --qubit q9 --simulate --no-save
python -m calibrations_v2.runner run resonator --load data/calibrations/2026-06-13/02a_resonator_spectroscopy/15-09-48-460578
```

Parameter overrides use `--set name=value`. Runtime lifecycle switches use
`--option name=value`, matching `CalibrationOptions`.

By default, profile updates may be staged but are not applied. Pass `--apply`
only when you explicitly want the runner to apply a proposed profile update.

Use `--dry-run` to print the resolved calibration, parameters, and options
without constructing a machine:

```powershell
python -m calibrations_v2.runner run resonator --dry-run --qubit q9 --set num_shots=50
```

The runner also accepts JSON recipes:

```json
{
  "calibration": "resonator",
  "qubit": "q9",
  "parameters": {
    "num_shots": 200,
    "frequency_span_in_mhz": 30
  },
  "options": {
    "plot_data": false
  }
}
```

Run a recipe with:

```powershell
python -m calibrations_v2.runner run --recipe path/to/recipe.json
```

`PowerRabi` is the first concrete v2 calibration:

```python
from calibration_utils.power_rabi import Parameters
from calibrations_v2.power_rabi import PowerRabi
from quam_config import create_machine

parameters = Parameters()
parameters.reset_type = "thermal"
parameters.num_shots = 500
parameters.transition = "ge"
parameters.pi_repetitions = 4

power_rabi = PowerRabi(
    parameters=parameters,
    machine=create_machine(qubit="q9"),
    auto_connect=True,
)
power_rabi.run()
```
