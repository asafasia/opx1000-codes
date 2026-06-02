# OPX1000 Calibration Codes

This repository contains QUA and QUAlibrate scripts for working with an OPX1000 setup.
It includes machine configuration helpers, calibration nodes, spectroscopy analysis utilities,
scope/live testing scripts, and a temperature monitor for controller diagnostics.

## Main Parts

- `quam_config/` defines and loads the QuAM machine configuration.
- `calibrations/` contains runnable calibration nodes such as hello QUA and qubit spectroscopy.
- `calibration_utils/` holds parameter, analysis, and plotting helpers used by the calibration nodes.
- `temperature_monitor/` logs controller temperatures and writes reports/plots under `data/`.
- `old/`, `my_quam_old/`, and `scope/` contain older examples and quick test scripts.

Generated measurement and monitoring outputs are written to `data/`, which is ignored by Git.
