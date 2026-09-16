# T2* Ramsey stability versus time

From the repository root in the lab Python environment:

```powershell
python -m sweeps.ramsey_stability --qubit q3 --duration-hours 24 --dry-run
python -m sweeps.ramsey_stability --qubit q3 --duration-hours 24
```

The second command executes real hardware. The runner repeats the existing
`calibrations/06a_ramsey.py` pulse sequence and analysis using a `Ramsey` subclass
of `BaseCalibration`. It builds one machine from the selected profile at startup.
It does not update frequencies, T2 metrics, profiles, or apply calibration proposals.
The existing output-inhibit latch, QM session ownership, and profile DC-bias
context remain in force. Profile bias is applied/restored per acquisition, as in
a standalone Ramsey calibration; this is not a continuous-bias hold experiment.

Set the Ramsey sweep for the qubit before an overnight run. Defaults match the
Ramsey parameter class (1,000 shots, thermal reset, analog readout, 16–3,000 ns,
50 wait points, 1 MHz virtual detuning). Those defaults are not an optimized
measurement recipe for every qubit. The delay range should resolve the oscillations
and the decay envelope; check a short run's traces and uncertainty first. Keep the
chosen recipe fixed during a stability run.

For example, with a previously validated 20 us delay range and discriminated readout:

```powershell
python -m sweeps.ramsey_stability --qubit q3 --duration-hours 24 `
  --interval-seconds 60 --num-shots 2000 --max-wait-ns 20000 `
  --wait-points 250 --detuning-mhz 1 --state-discrimination
```

`--interval-seconds` is the desired start-to-start period. Zero means back-to-back
acquisitions. If measurement, fitting, or saving overruns a slot, the runner skips
missed slots; it does not queue catch-up measurements. The monotonic duration
starts after setup. No new point starts at or after the deadline; an in-flight
point finishes and saves, so total runtime can exceed the requested duration by
that point's runtime. `--timeout-seconds` is the existing Ramsey/QM session timeout,
not a hard wall-clock deadline for acquisition, fitting, or the entire run.
`--max-points` optionally ends a short check early. Ctrl+C preserves partial results.

## Live results and recovery

Each invocation creates a unique folder under
`data/calibrations/YYYY-MM-DD/ramsey_stability/`. The terminal prints its path.
Open `index.html` in a browser; it refreshes every 15 seconds. Each completed attempt
updates `points.csv`, `summary.json`, and `stability.png`. The report shows T2* with
fit uncertainties, rolling variance, frequency offset, candidate jumps, and Allan
deviation where applicable. It also shows mean, median, standard deviation,
coefficient of variation, robust spread, peak-to-peak span, and linear drift.

The HTML and live chart are rewritten after every experiment; an open browser
reloads them every 15 seconds. Rejected experiments also save their Ramsey trace
and available fit to `points/NNNNNN/ramsey.png`, with the rejection reason. If
acquisition produced no data, the figure records that instead. Every 15th
experiment (15, 30, 45, ...) saves a Ramsey plot in its point folder and a
permanent copy of the T2* stability chart in `figures/stability_NNNNNN.png`.
These figures are saved without opening plot windows.

The durable measurement record is `points.jsonl` plus each `points/000001/point.json`.
Point JSON is atomically replaced and the journal is flushed/fsynced before the
next measurement. CSV, summaries, and plots are replaceable views. On abrupt process
or power loss they can lag the point files; ignore an incomplete last journal line
and recover from the numbered `point.json` files. A `started.json` without a
`point.json` marks an unfinished attempt. There is no automatic resume into an old
run; a new invocation creates a separate run to keep interruptions explicit.

Each point contains `raw.npz` and `raw.json`, saved before mitigation or analysis.
Successful analysis also saves `fit.npz`, `fit.json`, and `fit_results.json`.
Optional readout mitigation saves a separate `mitigated` dataset. NPZ files load
with `numpy.load(path, allow_pickle=False)`; companion JSON records dimensions,
coordinate names, and attributes. The initial run-level `sweep.npz` and `results.npz`
are empty setup placeholders from the standard saver; use `points.csv` for the
time series. Profiles, resolved machine/config (including kernels), parameters,
Git revision/dirty status, and Python version are recorded at startup.

No calibration objects, raw histories, or GUI figures accumulate in memory.
The runner retains only scalar rows. Disk write failures stop acquisition. Runtime
exceptions stop immediately after attempting to record the failure. Rejected fits
remain in the record and acquisition continues until
`--max-consecutive-failures` (default 15) is reached. Fix the cause before starting a
new run. Keep the computer awake and ensure disk capacity for the intended duration.

## Interpretation of diagnostics

- This measures **T2*** from Ramsey, not echo T2. Fit results in seconds are converted
  to explicit `t2_us`/`t2_error_us` columns; detuning offsets remain in Hz.
- Each row records UTC start/end, measurement-window UTC start/end, monotonic
  elapsed midpoint, measurement duration, and total point duration. The measurement
  window wraps Ramsey execution (including connection/session wait and fetching),
  excluding QUA construction, fitting, and file writes. It is not shot-level timing.
  A failure before execution falls back to the point window and has null measurement
  UTC fields. Allan spacing uses the measurement-window midpoints.
- Accepted points require fitter success, finite positive T2*, finite frequency
  offset, and a finite nonnegative fit uncertainty with relative error at most
  `--max-relative-error` (default 0.5). This is a configurable screening rule,
  not proof of model adequacy. Preserve and inspect rejected traces.
- Temporal variance is the unweighted **sample variance** (`ddof=1`); its square
  root is standard deviation. It includes estimator noise and physical variation.
  Rolling statistics use at most `--rolling-window` (default 20) consecutive accepted
  points; a failure breaks the window. Fewer than two points have no variance.
- Linear drift is ordinary least squares in us/hour against actual elapsed time.
  It is descriptive; no independence-based significance or confidence claim is made.
- Jump candidates compare each adjacent T2 change to the median historical change.
  After at least five preceding consecutive accepted points, the change is divided
  by the larger of the historical difference MAD scale (`1.4826 × MAD`) and the
  quadrature sum of adjacent fit errors. Values above `--jump-sigma` (default 5)
  are flagged. This is a causal screening heuristic, not a physical-event detector
  or calibrated false-alarm probability; covariance fit errors are approximate.
  Candidates remain in all accepted-point statistics. Failures are never bridged.
- Overlapping Allan deviation uses adjacent block means, averaging factors 1, 2,
  4, … and at least three differences per scale. It is withheld for any rejected
  point or midpoint spacing differing from the median by more than 5%; there is
  no gap filling or resampling. Pair counts are reported, not independent sample
  counts. The result is absolute deviation of the sampled T2 estimator in us,
  with no detrending or acquisition dead-time correction. It must not be treated
  as fractional oscillator-frequency stability or used alone to identify a noise
  mechanism. Irregular acquisitions can still use the time trace and other metrics.

The overlapping Allan estimator and care around gaps/dead time follow the methods
described in [NIST SP 1065, Handbook of Frequency Stability Analysis](https://www.nist.gov/publications/handbook-frequency-stability-analysis).
The T2 acceptance and jump thresholds are experiment-specific operational choices.

Run `python -m sweeps.ramsey_stability --help` for all options. Tests use synthetic
calibrations and clocks: `python -m pytest tests/test_ramsey_stability.py`.
