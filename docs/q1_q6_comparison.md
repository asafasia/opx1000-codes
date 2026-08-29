# Q1 and Q6 Parameter Comparison

Updated: 2026-08-29

This comparison uses the current `single_qubit` profile together with the latest successful T1 fits available for each qubit.

| Parameter | Q1 | Q6 |
|---|---:|---:|
| Qubit frequency, $f_{01}$ | 4.263630 GHz | 3.945142 GHz |
| EF transition frequency, $f_{12}$ | 4.050000 GHz | 3.707202 GHz |
| Anharmonicity | 217.11 MHz | **237.95 MHz** |
| Readout resonator frequency | 6.669000 GHz | 7.174350 GHz |
| Latest successful T1 | **50.18 +/- 1.34 us** | **12.10 +/- 0.18 us** |
| Ramsey T2* | 6.50 us | **13.19 us** |
| Echo T2 | 17.10 us | Not measured |
| Active pi pulse | Cosine, 40 ns | Constant, 40 ns |
| Active pi-pulse amplitude | 0.1458 | 0.2048 |
| Readout amplitude | 0.1000 | 0.2377 |
| Readout duration | 2 us | 3 us |
| Thermal readout fidelity | **86.66%** | 82.25% |
| Active-reset readout fidelity | 72.14% | **87.37%** |

## Main observations

- Q1 is approximately 318.49 MHz higher in $f_{01}$ than Q6.
- Q6 has approximately 20.84 MHz larger anharmonicity than Q1.
- Q1's latest measured T1 is approximately 4.15 times longer than Q6's.
- Q6's Ramsey T2* is approximately twice Q1's.
- Q1 has the better thermal readout fidelity, while Q6 has the better active-reset readout fidelity.
- An echo-T2 value is available for Q1 but not for Q6.

## Q6 anharmonicity update

The latest applied Q6 spectroscopy calibration set:

- $f_{01} = 3.945142098$ GHz
- $f_{12} = 3.707202098$ GHz
- Profile anharmonicity: **237.950 MHz**

The direct difference $f_{01}-f_{12}$ is 237.940 MHz. The 10 kHz difference from the stored anharmonicity occurred because $f_{01}$ was refined by -10 kHz immediately after the anharmonicity update.

## Data notes

- The latest successful Q1 T1 fit was measured on 2026-08-26: 50.1768 +/- 1.3370 us.
- The latest successful Q6 T1 fit was measured on 2026-08-24: 12.0999 +/- 0.1805 us.
- Ramsey T2* results are currently stored in seconds in fields named `t2_ramsey_ns`; the values above are converted to microseconds.
- The active pi pulses use different waveform shapes, so their numerical amplitudes should not be compared as equivalent pulse efficiencies.

## Sources

- `profiles/single_qubit/qubits.json`
- `profiles/single_qubit/pulses.json`
- `profiles/single_qubit/metrics.json`
- `data/calibration_updates/2026-08-29/03a_qubit_spectroscopy/14-17-39-261297/proposal.json`
- `data/calibrations/2026-08-26/05_T1/13-08-38-436517/analysis_result.json`
- `data/calibrations/2026-08-24/05_T1/23-17-20-472380/analysis_result.json`
