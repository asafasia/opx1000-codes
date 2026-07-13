# OPX1000 Lab Home

This local hub starts and links the four browser-based lab tools: Data Review,
Lab Monitor, Profile Studio, and Parameter Sweep.

From the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python apps/super_app/server.py
```

Open <http://127.0.0.1:8890>. Stopping the hub also stops any app processes it
started. Apps that were already running are detected and left alone.

The hub verifies each app's identity, writes child-process output to
`data/app_logs/super_app/`, and automatically restarts a crashed app up to four
times with backoff. A port occupied by an unrelated service is reported in the
UI instead of being mistaken for a healthy lab app.

The **Repository Wiki** card discovers all `.md` files under the repository and
renders them in a searchable reader. Generated `data/`, hidden directories,
dependency folders, and cache trees are deliberately excluded. Markdown links
between repository documents stay inside the wiki.

Use `--no-launch` to show the home screen without starting linked app servers.
