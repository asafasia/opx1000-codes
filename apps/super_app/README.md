# OPX1000 Quantum Coherence Lab

This local hub starts and links Data Review, Lab Monitor, Fridge Monitor,
Oscilloscope, Profile Studio, and Parameter Sweep. It runs as a native Windows desktop app
while keeping the services connected to the OPX1000 repository.

## Desktop app

Double-click `Quantum Coherence Lab.cmd` in the separate app directory. The launcher opens the
packaged `dist/Quantum Coherence Lab.exe` in a
dedicated **OPX1000 Quantum Coherence Lab** window with no browser tabs or address bar. Closing
the window stops the hub and any linked services that it started; services
that were already running are left alone.

The desktop shell uses `pywebview` with the Microsoft Edge WebView2 runtime.
To rebuild the executable, install the shell and build dependencies, then run
the checked-in build script:

```powershell
C:\Users\owner\miniconda3\envs\opx1000_env\python.exe -m pip install pywebview pyinstaller
powershell -ExecutionPolicy Bypass -File build_desktop.ps1
```

For diagnostics, the native app can also be launched from its own directory:

```powershell
C:\Users\owner\miniconda3\envs\opx1000_env\python.exe desktop.py --debug
```

## Browser development mode

From the separate app directory:

```powershell
python server.py
```

Open <http://127.0.0.1:8890>. Stopping the hub also stops any app processes it
started. Apps that were already running are detected and left alone.

The hub verifies each app's identity, writes child-process output to the app's
`logs/` directory when launched normally, and automatically restarts a crashed app up to four
times with backoff. A port occupied by an unrelated service is reported in the
UI instead of being mistaken for a healthy lab app.

The **Repository Wiki** card discovers all `.md` files under the repository and
renders them in a searchable reader. Generated `data/`, hidden directories,
dependency folders, and cache trees are deliberately excluded. Markdown links
between repository documents stay inside the wiki.

Use `--no-launch` to show the home screen without starting linked app servers.
