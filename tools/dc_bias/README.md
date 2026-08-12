# Arduino DC-bias control

This utility sends commands in the existing Arduino format:

```text
SET,<channel>,<voltage>\r
```

It supports channels 0 through 7 by default. After holding the requested bias
for the specified duration, it returns every channel to zero and closes the
serial connection. It also tries to zero all channels if the run is interrupted
or an error occurs after the connection is opened.

## Setup

Install the serial dependency in the active lab Python environment:

```powershell
python -m pip install -r tools/dc_bias/requirements.txt
```

## Usage

Preview the sequence without connecting to hardware:

```powershell
python tools/dc_bias/dc_bias.py --set 5=0 --set 7=0.06 --hold 2 --dry-run
```

Run the same sequence on the Arduino connected to `COM7`:

```powershell
python tools/dc_bias/dc_bias.py --port COM7 --set 5=0 --set 7=0.06 --hold 2
```

Use decimal volts: `0.06` means 60 mV. The utility does not know the safe
voltage range of the connected hardware, so verify the requested values before
running without `--dry-run`.
