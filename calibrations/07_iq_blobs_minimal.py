# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from calibration_utils.iq_blobs import Parameters
from saver import CalibrationSaver, current_profile_name
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits


description = """
Minimal fixed-frequency IQ-blobs acquisition.

For every shot, the experiment measures the ground-state IQ point, prepares the
qubit with the selected operation, and measures the prepared-state IQ point.
It preserves and saves the raw Ig, Qg, Im, and Qm points without analysis,
fitting, frequency sweeping, or state updates. A separate action plots the raw
IQ points after saving.
"""

node = QualibrationNode[Parameters, Quam](
    name="07_iq_blobs_minimal",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debugging parameter overrides."""
    node.parameters.qubits = ["q9"]
    node.parameters.qubit_operation = "saturation"


node.machine = create_machine()

node.machine.connect()  # Connect to the machine to fetch the qubits information and populate the node namespace if needed

node.machine.qmm.close_all_qms()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the fixed-frequency ground and prepared-state acquisition."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_runs = node.parameters.num_shots
    readout_operation = node.parameters.operation
    selected_qubit_operation = node.parameters.qubit_operation
    qua_qubit_operation = (
        "x180" if selected_qubit_operation == "x180_const" else selected_qubit_operation
    )
    if node.parameters.pi_repetitions < 1:
        raise ValueError("pi_repetitions must be a positive integer.")
    if node.parameters.xy_to_readout_delay_in_ns < 0:
        raise ValueError("xy_to_readout_delay_in_ns cannot be negative.")
    for qubit in qubits:
        if qua_qubit_operation not in qubit.xy.operations:
            raise ValueError(
                f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}."
            )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_runs), attrs={"long_name": "shot index"}),
    }

    with program() as node.namespace["qua_program"]:
        Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
        Im, Im_st, Qm, Qm_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                for qubit in multiplexed_qubits.values():
                    qubit.reset_qubit_thermal()
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(readout_operation, qua_vars=(Ig[i], Qg[i]))
                    save(Ig[i], Ig_st[i])
                    save(Qg[i], Qg_st[i])
                    qubit.reset_qubit_thermal()
                align()

                for qubit in multiplexed_qubits.values():
                    repetitions = (
                        node.parameters.pi_repetitions
                        if selected_qubit_operation == "x180_const"
                        else 1
                    )
                    for _ in range(repetitions):
                        qubit.xy.play(
                            qua_qubit_operation,
                            amplitude_scale=node.parameters.qubit_amplitude_factor,
                        )
                align()

                for qubit in multiplexed_qubits.values():
                    qubit.resonator.wait(node.parameters.xy_to_readout_delay_in_ns * u.ns)
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure(readout_operation, qua_vars=(Im[i], Qm[i]))
                    save(Im[i], Im_st[i])
                    save(Qm[i], Qm_st[i])
                    qubit.reset_qubit_thermal()
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ig_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Qg_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                Im_st[i].buffer(n_runs).save(f"Im{i + 1}")
                Qm_st[i].buffer(n_runs).save(f"Qm{i + 1}")


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the acquisition and store the untouched raw IQ points."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


@node.run_action()
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
    """Save only the raw Ig, Qg, Im, and Qm dataset."""
    output_directory = CalibrationSaver().save_xarray(
        node.name,
        node.results["ds_raw"][["Ig", "Qg", "Im", "Qm"]],
        profile_name=current_profile_name(),
    )
    node.namespace["calibration_run_directory"] = output_directory
    node.results["raw_data_directory"] = str(output_directory)
    node.log(f"Raw IQ data saved to {output_directory}")


# %% {Plot_raw_results}
@node.run_action()
def plot_raw_results(node: QualibrationNode[Parameters, Quam]):
    """Plot raw ground and prepared IQ points without blocking data saving."""
    ds = node.results["ds_raw"]
    figures = {}
    for qubit_name in ds.qubit.values:
        selected = ds.sel(qubit=qubit_name)
        figure, axis = plt.subplots()
        axis.scatter(selected.Ig, selected.Qg, s=4, alpha=0.3, label="Ground")
        axis.scatter(selected.Im, selected.Qm, s=4, alpha=0.3, label="Prepared")
        axis.scatter(
            float(selected.Ig.mean()),
            float(selected.Qg.mean()),
            s=120,
            marker="x",
            linewidths=3,
            label="Ground mean",
        )
        axis.scatter(
            float(selected.Im.mean()),
            float(selected.Qm.mean()),
            s=120,
            marker="x",
            linewidths=3,
            label="Prepared mean",
        )
        axis.set_title(str(qubit_name))
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.axis("equal")
        axis.legend()
        figure.tight_layout()
        figures[str(qubit_name)] = figure
    node.results["figures"] = figures
    plt.show(block=False)
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")

# %%
