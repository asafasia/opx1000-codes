# Calibration AI

Passive AI review helpers for completed calibration runs.

The package sends saved calibration figures to NVIDIA's
`nvidia/ising-calibration-1-35b-a3b` vision-language model and writes the
response beside the run as:

- `ai_review.json`
- `ai_review.md`

It does not modify the hardware profile or apply calibration updates.

## Hosted API

Set an NVIDIA API key and review a saved run:

```powershell
$env:NVIDIA_API_KEY = "<your key>"
python -m calibration_ai.review_run data\calibrations\2026-07-04\02a_resonator_spectroscopy\12-00-00-000000
```

## Local NIM

If the model is running locally as a NIM on port 8000:

```powershell
$env:NVIDIA_NIM_BASE_URL = "http://localhost:8000/v1"
python -m calibration_ai.review_run data\calibrations\2026-07-04\02a_resonator_spectroscopy\12-00-00-000000
```
