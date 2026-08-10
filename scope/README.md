# Live oscilloscope dashboard

This local web app displays real waveform data from the Tektronix oscilloscope
at `192.168.88.251`. It does not generate demo data and does not change the
scope's acquisition, trigger, vertical, or horizontal configuration.

The instrument command sequence matches `scope/live.py`. The complete waveform
is read from the scope, then downsampled locally for browser display.

From the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'C:\Users\owner\miniconda3\envs\opx1000_env\python.exe' scope\dashboard.py
```

Open <http://127.0.0.1:8765>. The app retries automatically while the scope's
VISA remote-control service is unavailable.

If the dashboard cannot open the VISA session, use the oscilloscope's Windows
desktop. First right-click the **TekVISA LAN Server Control** tray icon, stop
and restart VXI-11, and note any error dialog. If it still cannot be reached,
open Windows Firewall and allow TekVISA inbound traffic on the local network.
The dashboard reconnects without needing to be restarted.

Options:

```powershell
python scope\dashboard.py --scope-host 192.168.88.251 --port 8765 --timeout-ms 5000
```

For higher-rate acquisition, start **Socket Server** from the oscilloscope's
TekVISA LAN Server Control tray menu, then run:

```powershell
python scope\dashboard.py --socket-port 4000

The dashboard defaults to raw socket transport and never falls back to VXI-11.
```
