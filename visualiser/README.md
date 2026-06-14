# Data Review Dashboard

Local, read-only experiment browser for this repository. It scans `data/`, the
structured `data/calibrations/` archive, calibration proposals, and matching
scripts in `calibrations/`.

Run from the project root:

```powershell
python visualiser/server.py
```

Then open <http://127.0.0.1:8765>.

Optional:

```powershell
python visualiser/server.py --host 0.0.0.0 --port 9000
```

The server uses only the Python standard library. It is read-only, caps large
previews, and reports unreadable or malformed experiment files in the UI.
