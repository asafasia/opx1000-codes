import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "192.168.88.253"  # QOP IP address
port = None  # QOP Port
cluster_name = "katz_cluster"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[7])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [1]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for qubits 1 to 4 and 5 to 8 on two feed-lines
res_ch = mw_fem_spec(con=1, slot=7, in_port=1, out_port=1)
# individual xy drive for qubits 1 to 4 on MW-FEM 1

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout lines
connectivity.add_resonator_line(qubits=qubits)
# The xy drive lines
connectivity.add_qubit_drive_lines(qubits=qubits)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=False)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
# user_input = input("Do you want to save the updated QUAM? (y/n)")
# if user_input.lower() == "y":
machine = Quam()
#     # Build the wiring (wiring.json) and initiate the QUAM
build_quam_wiring(connectivity, host_ip, cluster_name, machine)

#     # Reload QUAM, build the QUAM object and save the state as state.json
# machine = Quam.load()
build_quam(machine)

# %%
