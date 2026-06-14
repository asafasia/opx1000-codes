# Wiring

## MW-FEM Shared LO Pairs

The following MW-FEM output-port pairs share an LO. When both outputs in a pair
are configured, they must use the same `lo_frequency_hz`:

| Shared-LO pair |
| --- |
| Outputs 2 and 3 |
| Outputs 4 and 5 |
| Outputs 6 and 7 |
| Outputs 8 and 9 |
| Outputs 10 and 11 |

This rule is enforced by the device-profile validator.

## Current Connections

The operational values are defined in
[`profiles/main/connectivity.json`](../../profiles/main/connectivity.json).

| Purpose | Qubits | Controller / FEM | Port | Band | LO |
| --- | --- | --- | --- | --- | --- |
| Shared XY drive | q7, q8 | con1 / 7 | Output 3 | 1 | 4.2 GHz |
| Shared XY drive | q9, q10 | con1 / 7 | Output 2 | 1 | 4.2 GHz |
| Shared resonator output | q7-q10 | con1 / 7 | Output 1 | 3 | 7.4 GHz |
| Shared resonator input | q7-q10 | con1 / 7 | Input 1 | 3 | 7.4 GHz |

Active qubits are selected separately in
[`profiles/main/profile.json`](../../profiles/main/profile.json). The table
above documents physical connections, including connections for inactive
qubits.
