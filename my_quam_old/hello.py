from qm import QuantumMachinesManager, SimulationConfig, qua
from old.configuration import cluster_name, qop_ip, qop_port

from qm.qua import *
from my_quam_old.machine import machine

qua_configuration = machine.generate_config()

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

n_avg = 100000  # The number of averages

with program() as prog:
    qubit = machine.qubits["q0"]

    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with strict_timing_():
            qubit.xy.play("const")
            align("qubit", "resonator")
            # I, Q = qubit.resonator.measure(
            #     "readout",
            # )
            qubit.resonator.wait(int(100e6))

            # I_st.save(I)  # Save the 'I' quadrature in the stream
            # Q_st.save(Q)  # Save the 'Q' quadrature in the stream

    # with stream_processing():
    #     # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
    #     I_st.buffer().average().save("I")
    #     Q_st.buffer().average().save("Q")


simulate = False  # Set to False to run on the OPX hardware, True to simulate


if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=3_000 // 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(qua_configuration, prog, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object


else:
    # Open the quantum machine
    qm = qmm.open_qm(qua_configuration)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(prog)
    # Get results from QUA program
