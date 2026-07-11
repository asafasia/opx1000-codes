# Cutoff Scan Matrix

This is the working plan for root-Lorentzian cutoff scans. The main idea is to
avoid using one scan grid for every cutoff. Broad lines need wide frequency
domains; narrow lines need finer frequency steps. Run coarse domains to avoid
clipping, then use finer domains only where the line is actually narrow enough
to justify them.

## Current Campaigns

| Campaign | Purpose | Cutoffs | Domains | Grid | Status |
| --- | --- | --- | --- | --- | --- |
| `20260711_overnight_400freq_200amp_100shots_run2` | Broad overnight map across high/medium/small regimes | default high, medium, small regions | high: 100/10/1 MHz; medium: 100/10/1 MHz; small: 100/10/1 MHz | 400 freq x 200 amp x 100 shots | high complete, small complete, medium failed near end of 1 MHz domain with QMM fetch error |
| `20260711_mid_001_to_01_10cutoffs_400freq` | Focused medium-cutoff rerun | 10 log-spaced cutoffs from `0.01` to `0.1` | 100/10/1/0.1 MHz | 400 freq x 200 amp x 100 shots | running |

## Scan Families

| Family | Cutoff Range | What Lines Usually Look Like | First Scan | Follow-Up Scan | Stop Condition |
| --- | --- | --- | --- | --- | --- |
| High cutoff | `0.1` to `0.99` | Square-pulse-like, often broad | `100 MHz`, 200-400 freq points, 100-200 amp points | `10 MHz`, then `1 MHz` only if not clipped and line is narrow | Feature is inside the domain and FWHM is much larger than the frequency step |
| Medium cutoff | `0.001` to `0.1` | Mixed: broad at some amplitudes, narrow at others | `100 MHz` and `10 MHz`, 400 freq points, 200 amp points | `1 MHz` or `0.1 MHz` near promising cutoffs/amplitudes | Same cutoff/amplitude gives consistent FWHM across adjacent domains |
| Small cutoff | `0.0005` to `0.001` | Potentially very narrow, but can still be clipped | `10 MHz`, 100-400 freq points, reduced amp grid if slow | `1 MHz`, `0.1 MHz`, maybe `0.01 MHz` | Narrow feature has several frequency samples across the linewidth |

## Domain Table

| Domain | Span | 400-Point Step | Best Use | Warning Sign |
| --- | ---: | ---: | --- | --- |
| `domain_100mhz` | `100 MHz` | `250.6 kHz` | Find broad lines, guard against features outside smaller windows | Useless for linewidth if the line is narrower than a few steps |
| `domain_10mhz` | `10 MHz` | `25.1 kHz` | Main comparison domain for medium/high cutoffs | If line touches edge, go back to `100 MHz` |
| `domain_1mhz` | `1 MHz` | `2.51 kHz` | Narrow-line follow-up | If line is broad or multi-feature, this domain can mislead |
| `domain_0p1mhz` | `0.1 MHz` | `251 Hz` | Very narrow final zoom | Only meaningful after a wider scan proves the line center is stable |
| `domain_0p01mhz` | `0.01 MHz` | `25 Hz` | Last-resort ultra-fine zoom | Expensive and risky unless center/frequency drift are controlled |

## Practical Decision Rules

Use the wide domain when:

- The peak or dip is cut off at the frequency edge.
- The fitted FWHM is a large fraction of the scan span.
- The center jumps between adjacent amplitudes.
- There are multiple features and the fit is choosing the wrong one.

Use the fine domain when:

- The line is centered inside the wider domain.
- The feature is only a few points wide in the wider scan.
- The best cutoff is already roughly known.
- You need a final linewidth estimate rather than discovery.

Increase shots when:

- The line is visible but noisy.
- The best cutoff changes mostly because of fit noise.
- The signal heatmap is weak but not flat.

Reduce amplitude points when:

- You only care about a known Rabi-frequency band.
- The broad scan already showed which amplitudes are useful.
- You want more cutoff/domain coverage without making the run too long.

## Recommended Scan Queue

| Priority | Run | Why |
| ---: | --- | --- |
| 1 | Medium `0.01..0.1`, 10 cutoffs, `100/10/1/0.1 MHz`, 400 freq x 200 amp x 100 shots | Current focused run; fills the important mixed region with enough resolution |
| 2 | Medium `0.001..0.01`, 8-10 cutoffs, `100/10/1 MHz`, 400 freq x 100-200 amp x 100 shots | Completes lower medium region without spending immediately on `0.1 MHz` |
| 3 | Small `0.0005..0.001`, 5-7 cutoffs, `10/1/0.1 MHz`, 400 freq x 50-100 amp x 100 shots | Fine-line search where high amplitude density is less important at first |
| 4 | Best-cutoff zooms, selected domains only | After summaries show promising cutoff/amplitude bands |

## Naming Convention

Use names that encode the important knobs:

```text
YYYYMMDD_<region>_<cutoff-range>_<cutoff-count>cutoffs_<freq-points>freq_<amp-points>amp_<shots>shots
```

Examples:

```text
20260711_mid_001_to_01_10cutoffs_400freq
20260712_mid_0001_to_001_8cutoffs_400freq_100amp_100shots
20260712_small_00005_to_0001_7cutoffs_400freq_100amp_100shots
```

## Files To Compare After Each Run

For each campaign, inspect:

```text
cutoff_sweep_summary.png
cutoff_sweep_fwhm_heatmap.png
cutoff_sweep_per_cutoff_traces.png
region_best_signal.csv
region_fit_results.csv
```

For a domain-level sanity check, open the `individual_figures/` folder and look
at the per-cutoff 2D heatmaps. Do not trust a narrow FWHM number when the raw
trace is edge-clipped, under-resolved, or visibly multi-feature.

## Follow-Up Template

Before starting a new scan, fill one row:

| Date | Region | Cutoffs | Domains | Freq Points | Amp Points | Shots | Why This Scan | Result / Next Step |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
|  |  |  |  |  |  |  |  |  |

