import pyvisa
import numpy as np
import matplotlib.pyplot as plt

rm = pyvisa.ResourceManager()

scope = rm.open_resource("TCPIP::192.168.88.251::INSTR")
scope.encoding = "latin-1"
scope.timeout = 10000

print(scope.query("*IDN?"))

plt.figure(figsize=(12, 6))

for ch in range(1, 5):

    channel = f"CH{ch}"

    try:
        # Select channel
        scope.write(f"DATA:SOURCE {channel}")
        scope.write("DATA:WIDTH 1")
        scope.write("DATA:ENC RPB")

        # Waveform scaling
        x_increment = float(scope.query("WFMOUTPRE:XINCR?"))
        x_zero = float(scope.query("WFMOUTPRE:XZERO?"))

        y_multiplier = float(scope.query("WFMOUTPRE:YMULT?"))
        y_zero = float(scope.query("WFMOUTPRE:YZERO?"))
        y_offset = float(scope.query("WFMOUTPRE:YOFF?"))

        # Acquire waveform
        scope.write("CURVE?")
        raw = scope.read_raw()

        # Parse binary block
        header_digits = int(raw[1:2])
        header_len = 2 + header_digits

        data_len = int(raw[2:header_len])

        data = raw[header_len:header_len + data_len]

        adc = np.frombuffer(data, dtype=np.uint8)

        # Convert to volts
        voltage = (adc - y_offset) * y_multiplier + y_zero

        # Time axis
        time = x_zero + np.arange(len(voltage)) * x_increment

        # Plot
        plt.plot(time, voltage, label=channel)

        print(f"{channel}: loaded {len(voltage)} points")

    except Exception as e:
        print(f"{channel}: failed -> {e}")

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Tektronix MSO70804C")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()