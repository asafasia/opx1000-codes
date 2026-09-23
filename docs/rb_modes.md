# Randomized benchmarking modes

Select `Parameters.mode = "standard"`, `"interleaved"`, or `"leakage"` for `rb`.
The parameter schema exposes these three choices to parameter forms.
`rb-interleaved` remains available and now uses the same implementation.

## Usage

These commands only inspect configuration; they do not execute hardware:

```powershell
python -m calibrations.runner run rb --qubit q3 --set mode=standard --dry-run
python -m calibrations.runner run rb --qubit q3 --set mode=interleaved --set interleaved_gate_operation=x90 --dry-run
python -m calibrations.runner run rb --qubit q3 --set mode=leakage --set 'readout_states=["g","e","f"]' --set use_state_discrimination=true --dry-run
```

In a Python run configuration:

```python
parameters.mode = "leakage"
parameters.readout_states = ["g", "e", "f"]
parameters.use_state_discrimination = True
parameters.fidelity_bootstrap_samples = 200
parameters.fidelity_bootstrap_seed = 42
```

Configure the existing `readout_GEF` pulse, GEF centers, and (when using
`use_readout_mitigation=True`) its full 3x3 assignment matrix first. Leakage
mode rejects GE-only or analog acquisition. Other modes retain GE, GEF and
analog readout support. GEF analysis uses P(g), never 1-P(e).
No frequencies, pulse calibrations, wiring, or readout kernels are altered by
selecting a mode. The existing `gate_family` selection applies to both IRB arms.

## Standard

Fit P(g,m)=A p^m+B. The error per Clifford is (1-p)/2. The existing
primitive-gate estimate divides that by 1.875; this assumes comparable weak
errors across this Clifford decomposition and is not a direct measurement of
an individual gate. Non-negligible leakage can invalidate the single-exponential
interpretation. See [Magesan, Gambetta and Emerson (2011)](https://arxiv.org/abs/1009.3639).

## Interleaved

For each random sequence, acquire reference and interleaved curves using the
same Clifford draws and depth list. Insert the selected Clifford after every
random Clifford, and recover the complete product including all insertions.
Depth counts random Cliffords, excluding the final recovery.

Fit both curves independently and report r_target=(1-p_interleaved/p_reference)/2,
not the interleaved decay divided by 1.875. The result includes the systematic
bound from Eq. 5, separately from statistical uncertainty. This estimate assumes
small gate dependence; leakage, drift and coherent errors may bias it. Negative
estimates are retained and marked unsuccessful, never silently clipped.
See [Magesan et al. (2012), Eqs. 4–5](https://arxiv.org/abs/1203.4550).

## Leakage

Acquire the usual Clifford-plus-recovery sequences with three-state readout.
Fit P_comp=P(g)+P(e)=A+B lambda^m. Report leakage L1=(1-A)(1-lambda)
and seepage L2=A(1-lambda), both per random Clifford; P(f)=1-P_comp.
The g/e/f populations and the leakage fit are plotted together.

These are population-model estimates: calibrated assignment errors matter,
including relaxation during measurement. The model assumes sufficiently
randomized computational/leakage coherences and effectively depolarized leakage
space. A single f level satisfies the latter, but population above f is not
resolved. Coherent oscillations or structured residuals call for a different model.
See [Wood and Gambetta (2018), Eqs. 9–17 and Sec. III.2](https://arxiv.org/html/1704.03081).

This implementation estimates leakage/seepage only; it does not infer a
leakage-corrected computational fidelity or divide rates by 1.875. Those fidelity
fields remain unavailable, and leakage results do not update gate-fidelity metrics.

## Reading the output

- `fit_results` contains per-qubit estimates, success status and diagnostic message.
- IRB includes both decays, target fidelity, bootstrap SD and systematic bounds.
- Leakage includes both rates, their bootstrap SDs, decay and fitted asymptote.
- Whole random-sequence curves are resampled, keeping paired IRB arms and all
  depths together. `bootstrap_successes` reports successful refits. SD is unavailable
  with fewer than two sequences or too few successful refits; it is not a
  systematic-error estimate or a confidence interval.
- At least four distinct depths are needed for the new fits. A flat curve cannot
  identify leakage and seepage separately; failed fits retain the acquired data.
- Extend the depth range until curvature/asymptotic population is measurable.
  Inspect residuals and repeatability before accepting a rate. Fit convergence
  alone does not establish that the model describes the experiment.
- Legacy interleaved-only data lacks a reference and is rejected for target-gate
  analysis. New datasets save `rb_mode`, target gate and gate-family metadata.
- Standard mode can propose average gate fidelity updates. Interleaved mode only
  updates its target's in-memory fidelity; leakage mode does neither. Profile
  application remains controlled by the run options.
