# Knowledge Graphs

This folder collects calibration knowledge graphs: human-readable decision
graphs that can later become automated calibration procedures.

Each graph should describe:

- The calibration problem.
- The starting assumptions.
- The decision nodes and branches.
- The measurements used to choose the next step.
- The confidence checks before applying profile updates.
- Ideas for automating the procedure.

## Graphs

- [Qubit frequency discovery](qubit_frequency_discovery.md): finding a qubit
  frequency when the stored value is only a rough guess.

## Automation Data

Automation runs should save human-browsable outputs under
[`automation_data`](automation_data/README.md). The compact `*.jsonl` files are
good for timelines and scripts; the automation data folder is where users
should open saved figures and per-run summaries to inspect what was found.

## Suggested Future Graphs

- Resonator frequency discovery from a cold start.
- Readout amplitude optimization.
- Integration weight and threshold optimization.
- Rabi amplitude discovery.
- Ramsey frequency correction.
- Active reset readiness.
- Full device bring-up from empty profile to first usable gates.
