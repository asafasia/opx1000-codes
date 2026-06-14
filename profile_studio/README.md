# Profile Studio

Local structured editor for the device configuration under `profiles/`. It
uses the same lab branding as the Data Review Dashboard and provides four
tabs: Profile, Qubits, Pulses, and Connectivity.

Run from the project root:

```powershell
python profile_studio/server.py
```

Then open <http://127.0.0.1:8766>.

Profile Studio edits only the four known JSON files inside an existing,
complete profile directory. Saves are atomic, must contain valid JSON, and are
rejected when the file changed on disk after it was loaded.
