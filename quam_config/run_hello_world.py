from qm import QuantumMachinesManager, SimulationConfig, qua
from qualibrate import QualibrationNode
from old.configuration import cluster_name, qop_ip, qop_port

from qm.qua import *
from quam_config import Quam

# qua_configuration = machine.generate_config()
node = QualibrationNode(
    name="02a_resonator_spectroscopy",  # Name should be unique
)

description = """
        Basic script to play with the QUA program and test the QOP connectivity.
"""


node = QualibrationNode(
    name="00_hello_qua", description=description, parameters=Parameters()
)

# qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

# n_avg = 100  # The number of averages

# node = QualibrationNode(
#     name="02a_resonator_spectroscopy",
#     # Name should be unique
# )


# with program() as prog:
#     qubit = machine.qubits["q9"]

#     I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

#     with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
#         with strict_timing_():
#             qubit.xy.play("saturation")
#             rr = qubit.resonator

#             align("q9.xy")


#             rr.measure("readout")

#             qubit.resonator.wait(int(100e6))

#         # save(I, I_st[0])  # Save the 'I' quadrature in the stream
#         # save(Q, Q_st[0])  # Save the 'Q' quadrature in the stream

#     # with stream_processing():
#     #     # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
#     #     I_st[0].buffer().average().save("I")
#     #     Q_st[0].buffer().average().save("Q")


# simulate = True  # Set to False to run on the OPX hardware, True to simulate


# if simulate:
#     # Simulates the QUA program for the specified duration
#     simulation_config = SimulationConfig(duration=60_000 // 4)  # In clock cycles = 4ns
#     # Simulate blocks python until the simulation is done
#     job = qmm.simulate(qua_configuration, prog, simulation_config)
#     # Get the simulated samples
#     samples = job.get_simulated_samples()
#     # Plot the simulated samples
#     samples.con1.plot()
#     # Get the waveform report object


# else:
#     # Open the quantum machine
#     qm = qmm.open_qm(qua_configuration)
#     # Send the QUA program to the OPX, which compiles and executes it
#     job = qm.execute(prog)
#     # Get results from QUA program

#     # print(job.result_handles)
