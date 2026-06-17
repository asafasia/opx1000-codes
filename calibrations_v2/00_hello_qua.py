"""Class-based v2 migration for 00_hello_qua."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from quam_config import Quam, create_machine
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibrate import NodeParameters

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions

description = """
        Basic script to play with the QUA program and test the QOP connectivity.
"""




# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
# Create the machine directly from profiles/main without loading state.json.




# %% {Create_QUA_program}
# %% {Simulate}
# %% {Execute}
# %% {Save_results}

class HelloQua(BaseCalibration[NodeParameters, Quam]):
    """v2 class migration for ``calibrations/00_hello_qua.py``."""

    def __init__(
        self,
        parameters: NodeParameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="00_hello_qua",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )
    def create_qua_program(self):
        node = self
        """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
        # Class containing tools to help handle units and conversions.
        u = unit(coerce_to_integer=True)
        # Get the active qubits from the node and organize them by batches
        node.namespace["qubits"] = qubits = get_qubits(node)

        amps = np.linspace(-1, 1, 110)
        # Register the sweep axes to be added to the dataset when fetching data
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "amplitude": xr.DataArray(
                amps, attrs={"long_name": "amplitude scale", "units": ""}
            ),
        }

        # node.namespace["sweep"]
        # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            a = declare(fixed)
            for multiplexed_qubits in qubits.batch():
                # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
                for qubit in multiplexed_qubits.values():
                    # node.machine.initialize_qpu(target=qubit)
                    qubit.xy.update_frequency(0)
                align()

                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(a, amps)):
                        for i, qubit in multiplexed_qubits.items():
                            # qubit.z.play("const", duration=qubit.xy.operations["x180"].length * u.ns)
                            qubit.xy.play("saturation", amplitude_scale=a)
                            qubit.reset_qubit_thermal()
                        align()

            with stream_processing():
                n_st.save("n")
            # This example doesn't save I/Q, adjust if needed
            # I_st[0].buffer(len(amps)).average().save("I1")
            # Q_st[0].buffer(len(amps)).average().save("Q1")


        return node.namespace.get("qua_program")
    def simulate_qua_program(self):
        node = self
        """Connect to the QOP and simulate the QUA program"""
        # Connect to the QOP
        qmm = node.machine.connect()
        # Get the config from the machine
        config = node.machine.generate_config()
        # Simulate the QUA program, generate the waveform report and plot the simulated samples
        samples, fig, wf_report = simulate_and_plot(
            qmm, config, node.namespace["qua_program"], node.parameters
        )
        # Store the figure, waveform report and simulated samples
        node.results["simulation"] = {
            "figure": fig,
            "wf_report": wf_report,
            "samples": samples,
        }
        plt.show()


    def execute_qua_program(self):
        node = self
        """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
        # Connect to the QOP
        qmm = node.machine.connect()
        # Get the config from the machine
        config = node.machine.generate_config()
        # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            # The job is stored in the node namespace to be reused in the fetching_data run_action
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
            # Display the progress bar
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            # Display the execution report to expose possible runtime errors
            print(job.execution_report())
        # Register the raw dataset
        node.results["ds_raw"] = dataset




if __name__ == "__main__":
    parameters = NodeParameters()

    options = CalibrationOptions()

    calibration = HelloQua(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q9"),
    )
    calibration.run()
