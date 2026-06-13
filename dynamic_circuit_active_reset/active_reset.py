# %% {Imports}
"""Active-reset experiment with separate ground and excited acquisitions."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm import SimulationConfig
from qm.exceptions import QMSimulationError
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, create_machine
from calibration_utils.iq_blobs import Parameters
from saver import CalibrationSaver, current_profile_name
from qualibration_libs.data import XarrayDataFetcher, convert_IQ_to_V
from qualibration_libs.parameters import get_qubits


# %% {Description}
description = """
        ACTIVE RESET - SEPARATE ACQUISITIONS
This experiment independently acquires ground-prepared and excited-prepared
shots. Every shot measures the initial IQ point, conditionally applies x180
when the measured I quadrature is above the readout threshold, and measures
the final IQ point to verify the reset.

The resulting dataset contains the before/after IQ clouds and thresholded
states for both preparations. This node does not update device parameters.
"""

node = QualibrationNode[Parameters, Quam](
    name="active_reset",
    description=description,
    parameters=Parameters(),
)

node.machine = create_machine()


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debugging parameter overrides."""
    # node.parameters.qubits = ["q9"]
    # node.parameters.num_shots = 1000
    # node.parameters.simulate = True


def make_active_reset_program(
    node: QualibrationNode[Parameters, Quam],
    preparation: str,
    n_runs: int | None = None,
):
    """Build one active-reset program for ground or excited preparation."""
    if preparation not in {"g", "e"}:
        raise ValueError("preparation must be 'g' or 'e'")

    u = unit(coerce_to_integer=True)
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    n_runs = node.parameters.num_shots if n_runs is None else n_runs
    readout_operation = node.parameters.operation
    selected_qubit_operation = node.parameters.qubit_operation
    qua_qubit_operation = "x180" if selected_qubit_operation == "x180_const" else selected_qubit_operation

    if node.parameters.pi_repetitions < 1:
        raise ValueError("pi_repetitions must be a positive integer.")
    if node.parameters.xy_to_readout_delay_in_ns < 0:
        raise ValueError("xy_to_readout_delay_in_ns cannot be negative.")
    for qubit in qubits:
        if qua_qubit_operation not in qubit.xy.operations:
            raise ValueError(f"{qubit.name} does not define qubit operation {qua_qubit_operation!r}.")

    with program() as qua_program:
        initial_i, initial_i_st, initial_q, initial_q_st, n, n_st = node.machine.declare_qua_variables()
        final_i, final_i_st, final_q, final_q_st, _, _ = node.machine.declare_qua_variables()
        initial_state = [declare(int) for _ in range(num_qubits)]
        final_state = [declare(int) for _ in range(num_qubits)]
        reset_applied = [declare(int) for _ in range(num_qubits)]
        initial_state_st = [declare_stream() for _ in range(num_qubits)]
        final_state_st = [declare_stream() for _ in range(num_qubits)]
        reset_applied_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                for qubit in multiplexed_qubits.values():
                    qubit.reset_qubit_thermal()
                align()

                if preparation == "e":
                    repetitions = (
                        node.parameters.pi_repetitions
                        if selected_qubit_operation == "x180_const"
                        else 1
                    )
                    for qubit in multiplexed_qubits.values():
                        for _ in range(repetitions):
                            qubit.xy.play(
                                qua_qubit_operation,
                                amplitude_scale=node.parameters.qubit_amplitude_factor,
                            )
                    align()

                for qubit in multiplexed_qubits.values():
                    qubit.resonator.wait(node.parameters.xy_to_readout_delay_in_ns * u.ns)
                for i, qubit in multiplexed_qubits.items():
                    threshold = qubit.resonator.operations[readout_operation].threshold
                    qubit.resonator.measure(
                        readout_operation,
                        qua_vars=(initial_i[i], initial_q[i]),
                    )
                    assign(initial_state[i], Cast.to_int(initial_i[i] > threshold))
                    assign(reset_applied[i], 0)

                align()
                for i, qubit in multiplexed_qubits.items():
                    threshold = qubit.resonator.operations[readout_operation].threshold
                    # Active reset: apply x180 only when the initial measurement is |1>.
                    with if_(initial_i[i] > threshold):
                        assign(reset_applied[i], 1)
                        qubit.xy.play("x180")

                for qubit in multiplexed_qubits.values():
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()
                for i, qubit in multiplexed_qubits.items():
                    threshold = qubit.resonator.operations[readout_operation].threshold
                    qubit.resonator.measure(
                        readout_operation,
                        qua_vars=(final_i[i], final_q[i]),
                    )
                    assign(final_state[i], Cast.to_int(final_i[i] > threshold))

                    save(initial_i[i], initial_i_st[i])
                    save(initial_q[i], initial_q_st[i])
                    save(final_i[i], final_i_st[i])
                    save(final_q[i], final_q_st[i])
                    save(initial_state[i], initial_state_st[i])
                    save(final_state[i], final_state_st[i])
                    save(reset_applied[i], reset_applied_st[i])
                for qubit in multiplexed_qubits.values():
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                initial_i_st[i].buffer(n_runs).save(f"initial_I{i + 1}")
                initial_q_st[i].buffer(n_runs).save(f"initial_Q{i + 1}")
                final_i_st[i].buffer(n_runs).save(f"final_I{i + 1}")
                final_q_st[i].buffer(n_runs).save(f"final_Q{i + 1}")
                initial_state_st[i].buffer(n_runs).save(f"initial_state{i + 1}")
                final_state_st[i].buffer(n_runs).save(f"final_state{i + 1}")
                reset_applied_st[i].buffer(n_runs).save(f"reset_applied{i + 1}")

    return qua_program


def acquire_preparation(
    node: QualibrationNode[Parameters, Quam],
    qmm,
    config: dict,
    preparation: str,
) -> xr.Dataset:
    """Execute one preparation in its own QM session and suffix its results."""
    node.log(f"Starting independent {preparation}-prepared active-reset acquisition")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_programs"][preparation])
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in fetcher:
            progress_counter(
                fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=fetcher.t_start,
            )
        node.log(f"{preparation}-prepared active reset:\n{job.execution_report()}")

    return dataset.rename(
        {
            name: f"{name}_{preparation}"
            for name in dataset.data_vars
            if name != "n"
        }
    )


def simulate_program(node, qmm, config, qua_program):
    """Simulate one representative shot and plot samples when available."""
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, qua_program, simulation_config)
    job.wait_until("Done", timeout=node.parameters.timeout)
    waveform_report = job.get_simulated_waveform_report()
    try:
        samples = job.get_simulated_samples()
        figure, axes = plt.subplots(nrows=len(samples.keys()), sharex=True, squeeze=False)
        for axis, controller in zip(axes.flat, samples.keys()):
            plt.sca(axis)
            samples[controller].plot()
            axis.set_title(controller)
        figure.tight_layout()
    except QMSimulationError:
        node.log("Could not pull simulated samples; showing the waveform report instead.")
        waveform_report.create_plot(samples=None, plot=True, save_path=None)
        figure = plt.gcf()
        samples = None
    return samples, figure, waveform_report


# %% {Create_QUA_programs}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Create independent ground-prepared and excited-prepared reset programs."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    n_runs = node.parameters.num_shots
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_runs), attrs={"long_name": "shot index"}),
    }
    node.namespace["qua_programs"] = {
        preparation: make_active_reset_program(node, preparation)
        for preparation in ("g", "e")
    }


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Simulate one representative shot for each preparation."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    simulations = {}
    for preparation in ("g", "e"):
        qua_program = make_active_reset_program(node, preparation, n_runs=1)
        _, figure, _ = simulate_program(node, qmm, config, qua_program)
        simulations[preparation] = {"figure": figure}
    node.results["simulation"] = simulations
    plt.show()


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Execute both preparations as separate jobs and merge their results."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    ground = acquire_preparation(node, qmm, config, "g")
    excited = acquire_preparation(node, qmm, config, "e")
    dataset = xr.merge([ground, excited])
    iq_names = [
        f"{stage}_{quadrature}_{preparation}"
        for preparation in ("g", "e")
        for stage in ("initial", "final")
        for quadrature in ("I", "Q")
    ]
    node.results["ds_raw"] = convert_IQ_to_V(dataset, node.namespace["qubits"], IQ_list=iq_names)


# %% {Save_raw_results}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
    """Save all before/after reset IQ and state results with a profile snapshot."""
    output_directory = CalibrationSaver().save_xarray(
        node.name,
        node.results["ds_raw"],
        profile_name=current_profile_name(),
    )
    node.namespace["calibration_run_directory"] = output_directory
    node.log(f"Active-reset results saved to {output_directory}")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired active-reset dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Calculate initial/final excited fractions and reset success."""
    ds = node.results["ds_raw"]
    summaries = {}
    for qubit_name in ds.qubit.values:
        selected = ds.sel(qubit=qubit_name)
        summaries[str(qubit_name)] = {}
        for preparation in ("g", "e"):
            initial_excited = float(selected[f"initial_state_{preparation}"].mean())
            final_excited = float(selected[f"final_state_{preparation}"].mean())
            reset_applied = float(selected[f"reset_applied_{preparation}"].mean())
            summaries[str(qubit_name)][preparation] = {
                "initial_excited_fraction": initial_excited,
                "final_excited_fraction": final_excited,
                "reset_applied_fraction": reset_applied,
                "ground_after_reset_fraction": 1 - final_excited,
            }
            node.log(
                f"{qubit_name} {preparation}-prepared: initial |1>={initial_excited:.3f}, "
                f"reset applied={reset_applied:.3f}, final |1>={final_excited:.3f}"
            )
    node.results["reset_summary"] = summaries


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot before/after IQ clouds and thresholded-state probabilities."""
    ds = node.results["ds_raw"]
    figures = {}
    for qubit_name in ds.qubit.values:
        selected = ds.sel(qubit=qubit_name)
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        for preparation, color in (("g", "tab:blue"), ("e", "tab:red")):
            axes[0].scatter(
                1e3 * selected[f"initial_I_{preparation}"],
                1e3 * selected[f"initial_Q_{preparation}"],
                s=4,
                alpha=0.35,
                color=color,
                label=f"{preparation}-prepared",
            )
            axes[1].scatter(
                1e3 * selected[f"final_I_{preparation}"],
                1e3 * selected[f"final_Q_{preparation}"],
                s=4,
                alpha=0.35,
                color=color,
                label=f"{preparation}-prepared",
            )

        axes[0].set_title("Before active reset")
        axes[1].set_title("After active reset")
        for axis in axes[:2]:
            axis.set_xlabel("I [mV]")
            axis.set_ylabel("Q [mV]")
            axis.axis("equal")
            axis.legend()

        x = np.arange(2)
        initial = [
            float(selected[f"initial_state_{preparation}"].mean())
            for preparation in ("g", "e")
        ]
        final = [
            float(selected[f"final_state_{preparation}"].mean())
            for preparation in ("g", "e")
        ]
        width = 0.35
        axes[2].bar(x - width / 2, initial, width, label="Before reset")
        axes[2].bar(x + width / 2, final, width, label="After reset")
        axes[2].set_xticks(x, ["g-prepared", "e-prepared"])
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Measured |1> fraction")
        axes[2].set_title("Active-reset result")
        axes[2].legend()

        figure.suptitle(f"{qubit_name}: separate active-reset acquisitions")
        figure.tight_layout()
        figures[str(qubit_name)] = figure

    node.results["figures"] = figures
    plt.show()
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Active-reset figures saved to {figures_directory}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the merged dataset, analysis summary, and figures."""
    node.save()
