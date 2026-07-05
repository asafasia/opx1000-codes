# Profile Studio

Local structured editor for the device configuration under `profiles/`. It
uses the same lab branding as the Data Review Dashboard and provides four
tabs: Profile, Qubits, Pulses, and Connectivity.

Run from the project root:

```powershell
python apps/profile_studio/server.py
```

Then open <http://127.0.0.1:8893>.

Profile Studio uses `8893` by default. If that port is already in use, stop the
old Profile Studio process before starting a new one, or pass an explicit
temporary `--port`.

Profile Studio edits only the four known JSON files inside an existing,
complete profile directory. Saves are atomic, must contain valid JSON, and are
rejected when the file changed on disk after it was loaded.

Both `main` and `single_qubit` are complete profiles and appear in the Profile
selector. Changes are written only to the selected profile; editing
`single_qubit` never modifies or reads values from `main`.
