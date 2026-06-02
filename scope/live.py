import pyvisa
import numpy as np
import matplotlib.pyplot as plt

rm = pyvisa.ResourceManager()
scope = rm.open_resource("TCPIP::192.168.88.251::INSTR")
scope.encoding = "latin-1"
scope.timeout = 10000

print(scope.query("*IDN?"))

# Select channel
scope.write("DATA:SOURCE CH1")
scope.write("DATA:WIDTH 1")
scope.write("DATA:ENC RPB")   # positive binary

# Read waveform scaling
x_increment = float(scope.query("WFMOUTPRE:XINCR?"))
x_zero = float(scope.query("WFMOUTPRE:XZERO?"))
y_multiplier = float(scope.query("WFMOUTPRE:YMULT?"))
y_zero = float(scope.query("WFMOUTPRE:YZERO?"))
y_offset = float(scope.query("WFMOUTPRE:YOFF?"))

# Read waveform data
scope.write("CURVE?")
raw = scope.read_raw()

# Parse binary block
header_digits = int(raw[1:2])
header_len = 2 + header_digits
data_len = int(raw[2:header_len])
data = raw[header_len:header_len + data_len]

adc = np.frombuffer(data, dtype=np.uint8)

# Convert to physical units
voltage = (adc - y_offset) * y_multiplier + y_zero
time = x_zero + np.arange(len(voltage)) * x_increment

print(f"Time range: {time[0]:.6f} s to {time[-1]:.6f} s")

# plt.figure(figsize=(10, 4))
# plt.plot(time, voltage)
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.title("Tektronix MSO70804C - CH1")
# plt.grid(True)
# plt.show()