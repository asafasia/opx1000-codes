from quam.components import BasicQuam, IQChannel, MWChannel, SingleChannel, pulses
from qm import qua
from quam.components.channels import (
    InOutMWChannel,
    MWFEMAnalogInputPort,
    MWFEMAnalogOutputPort,
)


from quam.examples.superconducting_qubits import Transmon, Quam
from quam.components.pulses import GaussianPulse, SquarePulse, SquareReadoutPulse

from qm.qua import *

machine = Quam()

transmon = Transmon(id="q0")

machine.qubits[transmon.name] = transmon

transmon.xy = MWChannel(
    id="qubit",
    LO_frequency=5e9,
    RF_frequency=10e6,
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1",
        fem_id="6",
        port_id=2,
        band=1,
        upconverter_frequency=5e9,
        sampling_rate=1e9,
        full_scale_power_dbm=0,
    ),
)

transmon.resonator = InOutMWChannel(
    id="resonator",
    LO_frequency=5e9,
    RF_frequency=100e6,
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1",
        fem_id="6",
        port_id=1,
        band=1,
        upconverter_frequency=5e9,
        sampling_rate=1e9,
        full_scale_power_dbm=10,
        # delay=100,
    ),
    opx_input=MWFEMAnalogInputPort(
        controller_id="con1",
        fem_id="6",
        port_id=1,
        band=1,
        sampling_rate=1e9,
        downconverter_frequency=5e9,
    ),
    time_of_flight=100,
)


gaussian_pulse = GaussianPulse(length=20, amplitude=1, sigma=3)
machine.qubits["q0"].xy.operations["X90"] = gaussian_pulse

const_pulse = SquarePulse(length=20, amplitude=1)
machine.qubits["q0"].xy.operations["const"] = const_pulse


readout_pulse = SquareReadoutPulse(length=1000, amplitude=1)
machine.qubits["q0"].resonator.operations["readout"] = readout_pulse

qubit = machine.qubits["q0"]


with qua.program() as prog:
    qubit = machine.qubits["q0"]
    with infinite_loop_():
        align("qubit", "resonator")

        qubit.xy.play("const")
        # I, Q = qubit.resonator.measure("readout")

        qubit.xy.wait(5000)

if __name__ == "__main__":
    machine.print_summary()
