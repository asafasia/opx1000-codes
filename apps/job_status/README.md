# QOP Job Status

Local live dashboard for open QMs and QOP jobs.

```powershell
python apps/job_status/server.py
```

Then open:

```text
http://127.0.0.1:8895
```

The API is read-only and uses `QuantumMachinesManager.get_jobs()` through
`calibrations.job_status`.
