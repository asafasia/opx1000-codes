# Calibrations

`calibrations` is the class-based replacement for the repeated
`@node.run_action` scripts in `calibrations/`.

Each new calibration should subclass `BaseCalibration` and keep the experiment
logic in ordinary methods:

```python
from calibrations import BaseCalibration


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

For real single-qubit executions, the shared lifecycle also reads the selected
qubit's `dc_bias_v` from the profile. A nonzero value is applied on the
hardcoded DC-bias channel 0 immediately before `execute_qua_program()` and is
returned to 0 V afterward, including when execution raises an exception.
Simulation, dry-run, loaded-data analysis, missing bias configuration, and an
exact `dc_bias_v` of 0 do not open the DC-bias serial connection.

Useful inherited helpers include `get_qubits()`, `execute_qua_program()`,
`simulate_qua_program()`, `save_raw_results()`, `save_arrays()`,
`save_figures()`, `save_qua_debug_script()`, and `propose_profile_update()`.

Runtime behavior can be controlled with `CalibrationOptions`:

```python
from calibrations import CalibrationOptions

options = CalibrationOptions(
    save_raw_data=False,
    save_figures=False,
    ai_review=False,
    plot_data=False,
    update_state=False,
    propose_profile_update=False,
)
```

Pass `options=options` into any calibration constructor.
Set `ai_review=True` to review saved figures with `calibration_ai` after
figure saving. The review is written into the run directory as
`ai_review.json` and `ai_review.md`, and a short status line is printed through
the calibration logger.

Qubit experiments also inherit `use_readout_mitigation=False`. Enable it only
with `use_state_discrimination=True`; the shared lifecycle then applies the
inverse 2x2 IQ-blobs fidelity matrix to `state` before analysis and plotting,
while retaining the measured values as `state_unmitigated`. For example:

```powershell
python -m calibrations.runner run power-rabi --qubit q9 --set use_state_discrimination=true --set use_readout_mitigation=true
```

The selected qubit must have a non-singular `readout.confusion_matrix` in its
profile, normally proposed by the IQ-blobs calibration.

## Terminal runner

The lightweight terminal wrapper is meant for Codex and quick lab use:

```powershell
python -m calibrations.runner list
python -m calibrations.runner describe resonator
python -m calibrations.runner run resonator --qubit q9 --set num_shots=200
python -m calibrations.runner run power-rabi --qubit q9 --simulate --no-save
python -m calibrations.runner run resonator --load data/calibrations/2026-06-13/02a_resonator_spectroscopy/15-09-48-460578
python -m calibrations.runner run resonator --qubit q9 --option ai_review=true
```

Parameter overrides use `--set name=value`. Runtime lifecycle switches use
`--option name=value`, matching `CalibrationOptions`.

### Runtime estimates

Every class-based calibration reports its normalized sweep workload before the
QM is opened. When comparable saved runs exist, it also scales their measured
execution times to print an approximate duration. During execution, the progress
line replaces that historical estimate with an adaptive ETA after the first outer
iteration completes. Completed runs save `execution_duration_s` and the workload
estimate in `metadata.json`, so later estimates improve automatically.

The estimate is intentionally approximate: active reset, data transfer, dynamic
control flow, and changed pulse durations can alter the rate. Disable only the
pre-run message when needed with:

```powershell
python -m calibrations.runner run resonator --qubit q9 --option report_runtime_estimate=false
```

Code using a calibration object can inspect the estimate after building the QUA
program:

```python
calibration.namespace["qua_program"] = calibration.create_qua_program()
estimate = calibration.estimate_runtime()
print(estimate.estimated_seconds, estimate.workload_units)
```

By default, profile updates may be staged but are not applied. Pass `--apply`
only when you explicitly want the runner to apply a proposed profile update.

Use `--dry-run` to print the resolved calibration, parameters, and options
without constructing a machine:

```powershell
python -m calibrations.runner run resonator --dry-run --qubit q9 --set num_shots=50
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
python -m calibrations.runner run --recipe path/to/recipe.json
```

## Calibration-results database

Each completed command-runner calibration is registered in the local SQLite
database at `data/calibration_results.sqlite`. It contains run provenance,
outcomes, raw-data/figure paths, and staged profile-update records; raw arrays
remain in the calibration run directory. Initialise it explicitly (optional;
the runner creates it when needed) and query metric history with:

```powershell
python -m calibrations.results_db init
python -m calibrations.results_db history q3 t1_ns
```

Calibration implementations can record accepted fit values after a run:

```python
database.record_metric(run_id, target_name="q3", metric_name="t1_ns",
                       value=12345.0, uncertainty=120.0, unit="ns", accepted=True)
```

`PowerRabi` is the first concrete calibration:

```python
from calibration_utils.power_rabi import Parameters
from calibrations.power_rabi import PowerRabi
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
