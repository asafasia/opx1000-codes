# USB VISA bridge

This bridge allows the control PC (`192.168.88.250`) to issue a restricted set
of read-only VISA queries through the Windows laptop (`192.168.88.247`) that
owns the analyzer's USB connection.

## Laptop setup

1. Install the analyzer manufacturer's USB/VISA driver. NI-VISA or Keysight IO
   Libraries are common choices.
2. Install PyVISA:

   ```powershell
   py -m pip install pyvisa
   ```

3. Copy `tools/visa_usb_bridge.py` to the laptop and run:

   ```powershell
   py .\visa_usb_bridge.py
   ```

4. If Windows Defender Firewall asks, allow Python only on **private networks**.
5. Leave the terminal open and copy the random token it prints.

The bridge binds only to `192.168.88.247`, accepts only the control PC at
`192.168.88.250`, requires the random token, serializes VISA access, and has no
endpoint for write commands. It filters discovery to USB resources and rejects
TCP/IP, serial, and GPIB targets.

## Control-PC checks

Run these from the repository root, replacing `TOKEN` with the token printed on
the laptop:

```powershell
python .\tools\visa_usb_client.py --token TOKEN health
python .\tools\visa_usb_client.py --token TOKEN resources
python .\tools\visa_usb_client.py --token TOKEN query 'USB0::...::INSTR' '*IDN?'
```

Use the exact resource string returned by `resources`. The initial allowlist is
limited to instrument identity, options, SCPI version, and error-queue queries.
Analyzer-specific read-only configuration and trace queries should be added only
after the model is identified from `*IDN?`.

Press `Ctrl+C` in the laptop terminal to stop the bridge.
