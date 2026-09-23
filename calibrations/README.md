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
        return self.readout_profile_updates(frequency="frequency")
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
hardcoded DC-bias channel 0 (physical output 1) immediately before `execute_qua_program()` and is
returned to 0 V afterward, including when execution raises an exception.
Simulation, dry-run, loaded-data analysis, missing bias configuration, and an
exact `dc_bias_v` of 0 do not open the DC-bias serial connection.

Useful inherited helpers include `get_qubits()`, `execute_qua_program()`,
`simulate_qua_program()`, `save_raw_results()`, `save_arrays()`,
`save_figures()`, `save_qua_debug_script()`, and `propose_profile_update()`.

Profile update helpers keep calibration methods focused on the fitted values:

```python
# Absolute pulse amplitude from fit_results[qubit.name]["opt_amp"]:
return self.pulse_profile_updates("x180", amplitude="opt_amp")

# Multiply the dedicated profile pulse amplitude by a fitted factor:
return self.pulse_profile_updates("x180", amplitude_scale="optimal_amp_prefactor")

# Require a DRAG pulse and map the fitted alpha to the profile's beta field:
return self.pulse_profile_updates(self.parameters.operation, pulse_type="drag", beta="alpha")

# Update the selected GE/GEF readout and clear its stale centers/confusion matrix:
return self.readout_profile_updates(frequency="optimal_frequency", amplitude="optimal_amplitude")

# Relative fields in qubits.json or metrics.json:
return self.qubit_profile_updates({"frequencies_hz.qubit_f01": "frequency"})
return self.metric_profile_updates({"coherence.t2_ramsey_ns": "decay"})
```

Strings select keys from each qubit's `fit_results`; a callable `(qubit, fit)`
can compute a value, and numbers, booleans, lists, and `None` are literal values.
The helpers select successful qubits, resolve profile paths, and preserve the
0.7 V pulse-amplitude limit. Drive pulse updates require an explicit operation
mapping; derived aliases never silently update a parent pulse. Readout updates
use the selected pulse and mode. Custom calculations can iterate over
`self.profile_update_results()`.

These helpers only return proposed values. The inherited `propose_profile_update()`
handles staging and confirmation, including `apply=False`; calibration scripts
should normally override `profile_updates()` rather than repeat staging code.

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

### Spectroscopy versus external DC bias

`03d_qubit_spectroscopy_vs_external_flux.py` combines the spectroscopy map and
peak extraction from `03b` with a robust local parabola fit and the host-controlled outer loop from `03c`.
It uses the profile's Arduino DC-bias source on channel 0 (physical output 1). Select exactly one
qubit; an OPX Z line is not required. The original scripts remain available.

Preview parameters without connecting to either device:

```powershell
python -m calibrations.runner run qubit-external-flux --profile single_qubit --qubit q3 --set flux_offset_span_in_v=0.002 --set num_flux_points=11 --dry-run
```

Remove `--dry-run` when ready to acquire on the hardware. The important settings are:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `flux_bias_center_in_v` | Absolute voltage at the center; `None` uses the selected qubit's profile `dc_bias_v` | `None` |
| `flux_offset_span_in_v` | Total voltage span around that center | 0.002 V |
| `num_flux_points` | Number of external voltage settings, including endpoints | 11 |
| `bias_settle_time_s` | Host settling delay after setting each voltage | 0.1 s |
| `pause_timeout_s` | Timeout waiting for each pause or I/Q result pair | 300 s |
| `num_shots` | Averages at each voltage | 50 |
| `frequency_span_in_mhz` / `frequency_step_in_mhz` | Detuning sweep around the profile qubit frequency | 100 / 0.5 MHz |

For example, a profile bias of 0.003 V and span of 0.002 V sweeps from 0.002 to
0.004 V. All points must satisfy `dc_bias.max_abs_voltage_v` in the selected
profile's `connectivity.json`, which is the sole source of the voltage limit.
Pulse operation, amplitude factor and duration use the usual spectroscopy parameters.
This calibration uses thermal reset and IQ readout.

One QUA job pauses before each voltage point. Python sets the source, waits for
settling, resumes the job, and fetches an independently averaged frequency row.
Results use `save_all` and indexed fetching so asynchronous I/Q streams cannot
mix different voltage points; see the
[QM stream-processing documentation](https://docs.quantum-machines.co/latest/docs/Guides/stream_proc/).
A final pause has no extra measurement. Cleanup halts the job and attempts to
return the source to zero, including on exceptions. Simulation omits the pauses
and never connects to the bias source; it checks the pulse sequence, not the
external source or the physical frequency response.

The external bias remains applied during reset, spectroscopy, and readout.
The resonator frequency is held at its profile setting throughout the scan;
large bias changes may therefore need separate readout characterization.
Saved maps use absolute source voltage in volts. Analysis extracts spectroscopy
peaks independently from I and Q and fits each with a local parabola using a trimmed-residual initialization, outlier
rejection, and a refit to the consistent peaks. The vertex gives the extremum
voltage and fitted frequency, rather than selecting the highest measured point.
The valid I or Q fit with the higher R² on retained inlier peaks supplies the
extremum and profile-update proposal. Both candidates, point counts, and R²
scores (inlier and all-peak) are retained. Stacked I/Q plots show their own
parabolas, accepted/excluded peaks, R², and the selected quadrature.
At least five consistent peaks, significant curvature, and two inliers on each
side of the extremum are required. Flat data, uncertain or unbracketed extrema,
and extrema outside the measured frequency range do not produce profile updates.
The voltage uncertainty is a local fit estimate, not a hardware accuracy claim.
Successful fits stage `dc_bias_v` and `qubit_f01`; they are not applied by default.
Use a scan around one extremum: a parabola does not model a full flux period.

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


## AC Stark shift from the power Rabi chevron

`04d_power_rabi_chevron.py` (`power-rabi-chevron` in the runner) now extracts
one spectral maximum per amplitude and fits
`peak_detuning_hz = offset_hz + c * amplitude**2`, with a free offset.
It also fits `offset + k * abs(amplitude)**p` to test whether the measured
exponent is consistent with 2. It reports an inconclusive result when the
shift or exponent is unresolved. Edge, weak, split and unstable peaks are
rejected, and every decision is saved in `stark_peaks.csv`.

Use a plain square `saturation` operation for the weak-drive comparison.
A short coherent Rabi pulse can have two off-resonant maxima; fitting its
brightest fringe does not reliably measure a Stark shift. IQ data are analyzed
using the magnitude of the complex displacement from the off-resonant
baseline, so a dip or rotated readout response works too.

For a calibrated reference pi pulse and linear drive gain, the analysis uses
its actual sampled in-phase area to convert amplitude to Rabi frequency.
This handles cosine/Gaussian references rather than assuming a square pulse.
The reference is selected by `stark_reference_pi_operation` (default `x180`).
For a DRAG reference this area conversion is approximate. Verify the reference
calibration and configured anharmonicity before interpreting the coefficient;
fit error bars exclude their systematic uncertainties.

With `f_R = Omega/(2*pi)` and `Delta_f = f12-f01 < 0`, the weak, uncorrected
square-drive prediction is
`shift_hz = -f_R**2/(2*Delta_f) = C*f_R**2/abs(Delta_f)`, with `C = 0.5`.
All frequencies in this comparison are Hz; no additional `2*pi` is needed.
See [Chen, thesis chapter 7, Eqs. 7.1-7.5](https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf)
for the perturbative level shift, accounting for the factor-of-two Rabi
Hamiltonian convention. Shaped/DRAG scan pulses retain an empirical fit, with
no universal 0.5 comparison. A single fixed-anharmonicity scan tests quadratic
amplitude scaling, but cannot independently establish inverse-anharmonicity
scaling; that needs calibrated measurements at several anharmonicities.

A starting configuration for a finer scan is below. This command is a dry run;
check the selected qubit's existing saturation amplitude, pulse length and
calibrated pi reference before removing `--dry-run` to acquire data. The
amplitude prefactors below multiply the existing pulse amplitude. A 0.1 MHz
frequency step is more useful for sub-MHz shifts than the old 1-2 MHz scans.

```powershell
python -m calibrations.runner run power-rabi-chevron --qubit q1 --dry-run `
  --set operation=saturation --set frequency_span_in_mhz=40 `
  --set frequency_step_in_mhz=0.1 --set min_amp_factor=0 `
  --set max_amp_factor=0.6 --set amp_factor_step=0.02 --set num_shots=500 `
  --set stark_peak_window_mhz=18 --set fit_stark_shift=true `
  --option update_state=false --option propose_profile_update=false `
  --option apply_profile_update=false
```

Each amplitude trace is fitted to a positive Gaussian plus a linear background
using bounded, multistart `scipy.optimize.least_squares` with `soft_l1` loss.
Prominence and half-height widths supply center/width guesses; a median of up to
five neighboring amplitude traces supplies additional guesses only. Each final
fit uses its own unsmoothed measurements, without enforcing a smooth or
quadratic center trajectory. The full frequency scan constrains the background
and broad tails; `stark_peak_window_mhz` now bounds the fitted center, rather
than cropping the fitted samples. `None` allows a center anywhere in the scan.
The extracted frequency is the Gaussian center. Its error is a local robust
Jacobian covariance approximation, inflated by residual MAD with a 0.1-bin
floor; it is not a bootstrap confidence interval. Unresolved, edge-truncated,
and competing peaks are rejected. `stark_peaks.csv` also records fitted width,
height, background, significance, and the winning initial center and width,
including rejected candidate fits where available. More accepted peaks alone
do not establish a physical Stark shift.
`stark_min_peak_snr` is the fitted Gaussian height divided by its standard error.
`stark_peak_fit_points` (odd) controls minimum sample count and competing-peak
separation; it no longer limits the fit to a few samples around the maximum.
Other fit controls include
`stark_min_amp_factor`, `stark_max_amp_factor`,
`stark_max_rabi_to_anharmonicity` (default 0.2), and
`stark_min_valid_points` (default 6). Set `fit_stark_shift=false` for the
original heatmap-only analysis. Fit coefficients, uncertainties, exponent and
consistency verdict are in `analysis_result.json`; the chevron overlay and
`ac_stark_shift_<qubit>` figure show the extracted peaks and scaling test.
The experiment makes no profile or pulse updates.


## Choosing two-state or three-state readout

Use the same parameter in qubit experiments and IQ/readout calibrations:

```python
parameters.readout_states = ["g", "e", "f"]  # use ["g", "e"] for GE
parameters.use_state_discrimination = True
parameters.reset_type = "active"
parameters.active_reset_max_attempts = 15
```

The mode selects `readout` or `readout_GEF`, each with independent frequency,
amplitude, duration, weights, centers, and confusion matrix. Active reset uses
the same pulse and basis: E -> G in GE mode, E -> G or F -> E -> G in GEF mode.
GEF reset requires an `EF_x180` operation. Thermal reset remains available and
is required for the first calibration of an uncalibrated mode.

For the first GEF IQ calibration, select `readout_states=["g", "e", "f"]` and
`reset_type="thermal"` in `iq-blobs` (or use `iq-blobs-gef`). The standard
`iq-blobs` node acquires all three clouds automatically in this mode. Review
and apply its profile proposal before using GEF discrimination. The existing
q1/q6 GE centers do not calibrate their new GEF pulses. Frequency/amplitude
optimizers can tune either pulse; the two-cloud optimizers score G/E contrast,
while `gef-readout-frequency` scores three-state separation. The sliced-weight
optimizer still optimizes G/E contrast, with a separate kernel file per pulse.
Recalibrate all three IQ clouds after changing a GEF pulse or kernel.

A runner dry-run example (no hardware connection):

```powershell
python -m calibrations.runner run t1 --profile single_qubit --qubit q6 --set 'readout_states=["g","e","f"]' --set use_state_discrimination=true --set reset_type=active --dry-run
```

Discriminated datasets contain `population_g`, `population_e`, and, in GEF
mode, `population_f`. All are shot fractions. `state` remains an alias for
`population_e`, so existing fits and plots retain their P(e) meaning; it is
never an average of the integer labels 0/1/2. Inspect `population_f` to see
leakage. Randomized benchmarking uses measured P(g), so f leakage is not
counted as ground. Readout mitigation uses the selected pulse's full 2x2 or 3x3 matrix and
retains unmitigated populations. Raw IQ acquisition also uses the selected
pulse and its correct length for voltage conversion. Saved metadata includes
the readout basis and pulse name.

Select the mode through `readout_states`; IQ/weight calibration's legacy
`operation` field is resolved from that selection at program construction.
For IQ blobs, leave `states=None` (the default) to follow each mode switch
automatically. The `states` field on IQ/resonator spectroscopy describes prepared clouds;
it is distinct from the readout basis. Dedicated raw-IQ diagnostics remain
raw-IQ diagnostics even when the GEF pulse is selected.

## Shared averaged and single-shot acquisition

The class-based measurement calibrations share acquisition handling in
`BaseCalibration`. Set the mode in a script's settings or with the runner's
`--set acquisition=single_shot` option:

```python
parameters.acquisition = "averaged"     # controller averaging (default)
# parameters.acquisition = "single_shot"  # retain every measurement
parameters.use_state_discrimination = True  # False retains analog I/Q instead
parameters.readout_states = ["g", "e", "f"]  # optional calibrated GEF readout
```

`num_shots` is the number of repetitions per sweep point. In single-shot mode,
the raw dataset retains a `shot` dimension. For discrimination it contains
integer labels (0=g, 1=e, 2=f); for analog readout it contains individual I/Q
values. Analysis retains these as `state_shots` or `I_shots` / `Q_shots`, and
calculates the populations or mean IQ used by existing fits and plots. GEF
populations are separate label fractions; integer labels are never averaged.
Readout mitigation runs after population reduction and preserves the raw shots.
Saved xarray dimension names and metadata allow unambiguous reloading.

Shot position follows the experiment's loop order: ordinary sweeps have
`(qubit, shot, ...sweeps)`, whereas RB has
`(qubit, nb_of_sequences, [rb_variant,] depths, shot)`. Each RB sequence stays
separate for fitting and uncertainty estimation. Single-shot results are retained
after each completed outer iteration (one sweep, cloud shot, or RB sequence),
and storage scales with `num_shots`. The progress counter remains live.
For measurements using `BaseCalibration.fetch_result_dataset()`, Ctrl+C stops the
job and continues saving, analysis, and plotting with the completed measurements.
Single-shot data keeps only the common completed prefix across measurement streams;
missing shots are never filled with zeroes. Interrupted datasets are marked in
metadata and do not automatically update device state or profiles. If no complete
sweep exists yet, the run exits with a message. A second Ctrl+C aborts recovery.
Actual controller failures and data loss still raise errors. The repository's
`utils.qm_session` wrapper preserves unhandled interrupts after QM cleanup.
ADC time-of-flight measurements also support single-shot traces.

IQ blobs, resonator spectroscopy with cloud-fidelity analysis, readout-power
optimization, and readout-frequency/amplitude optimization require individual
IQ samples. Their setting is fixed to `single_shot`, preserving their existing
`n_runs` dimension and distribution analyses. The older procedural scripts
`03b` and `03c` do not use the class-based acquisition API.

When writing an experiment, define `namespace["sweep_axes"]` in outer-to-inner
loop order, then call `self.configure_acquisition()` before declaring streams.
The default shot loop is outermost; use an explicit `loop_order` including
`"shot"` when it is nested. Use `declare_state_stream()` and
`save_readout_state()` for discrimination, and finalize each qubit with:

```python
self.process_readout_streams(
    i, state_st if self.parameters.use_state_discrimination else None, I_st, Q_st
)
```

Named analog channels use `save_acquisition_stream(stream, result_name)`.
Call `prepare_acquisition_results()` at the start of an overridden
`analyse_data()` method so direct analysis calls also reduce shots correctly.
The base lifecycle does this automatically before mitigation and analysis.

## Randomized benchmarking modes

`rb` accepts `mode=standard`, `mode=interleaved`, or `mode=leakage`.
See [RB modes and analysis](../docs/rb_modes.md) for configuration, formulas,
uncertainties, assumptions and primary literature.
