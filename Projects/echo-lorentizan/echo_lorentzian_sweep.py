# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits

from calibration_io import CalibrationSaver, current_profile_name
from quam_config import Quam, create_machine
from utils.plotting_settings import plot_per_qubit
from utils.simulation import simulate_and_plot

from lorentzian import install_lorentzian_operation, plot_raw_data, process_raw_dataset
from parameters import Parameters

description = """
        ECHO LORENTZIAN - FREQUENCY VS AMPLITUDE
This project plays a fixed-length Lorentzian-like qubit pulse while sweeping both
the qubit-drive detuning and the Lorentzian amplitude. It follows the same
two-dimensional structure as the power Rabi chevron, but replaces the square or
DRAG operation with a user-length waveform pulse. The standard Lorentzian is:

    A / (1 + (t / tau)^2)

The root-Lorentzian option uses:

    A / sqrt(1 + (t / tau)^2)

where t is centered on the pulse midpoint and tau is derived from the requested
edge cutoff.
"""


node = QualibrationNode[Parameters, Quam](
    name="echo_lorentizan",
    description=description,
    parameters=Parameters(),
    machine=create_machine(),
)


node.machine.connect()
node.machine.qmm.close_all_qms()


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter overrides when running this file directly."""
    # node.parameters.qubits = ["q3"]
    # node.parameters.lorentzian_length_in_ns = 80
    # node.parameters.pulse_shape = "root_lorentzian"
    # node.parameters.lorentzian_tau_in_ns = 12
    # node.parameters.root_lorentzian_cutoff = 0.2
    # node.parameters.lorentzian_peak_amplitude = 0.08
    pass


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    variables = set(ds.data_vars)
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - variables
    present_unexpected = unexpected & variables
    if missing or present_unexpected:
        raise RuntimeError(
            "Echo-Lorentzian readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(variables)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the detuning-versus-amplitude Lorentzian sweep program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    operation = node.parameters.operation
    install_lorentzian_operation(node)

    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    if amps.size == 0:
        raise ValueError("Amplitude sweep is empty.")
    if np.any(np.abs(amps) >= 2):
        raise ValueError("QUA amplitude prefactors must stay within [-2, 2).")

    span = int(round(node.parameters.frequency_span_in_mhz * u.MHz))
    step = int(round(node.parameters.frequency_step_in_mhz * u.MHz))
    if step <= 0:
        raise ValueError("frequency_step_in_mhz must be positive.")
    dfs = np.arange(-span // 2, span // 2 + step, step, dtype=int)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
        ),
        "amp_prefactor": xr.DataArray(
            amps,
            attrs={"long_name": "Lorentzian amplitude prefactor"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(a, amps)):
                        for qubit in multiplexed_qubits.values():
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                            )
                            qubit.xy.update_frequency(
                                qubit.xy.intermediate_frequency + df
                            )
                        align()

                        for qubit in multiplexed_qubits.values():
                            qubit.xy.play(operation, amplitude_scale=a)
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(
                                    "readout", qua_vars=(I[i], Q[i])
                                )
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                        f"I{i + 1}"
                    )
                    Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(
                        f"Q{i + 1}"
                    )


@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, figure, waveform_report = simulate_and_plot(
        qmm,
        config,
        node.namespace["qua_program"],
        node.parameters,
    )
    node.results["simulation"] = {
        "figure": figure,
        "waveform_report": waveform_report,
        "samples": samples,
    }


@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in fetcher:
            progress_counter(
                fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=fetcher.t_start,
            )
        node.log(job.execution_report())
    validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
    node.results["ds_raw"] = dataset


@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def save_raw_results(node: QualibrationNode[Parameters, Quam]):
    output_directory = CalibrationSaver().save_xarray(
        node.name,
        node.results["ds_raw"],
        profile_name=current_profile_name(),
    )
    node.namespace["calibration_run_directory"] = output_directory
    node.log(f"Raw calibration results saved to {output_directory}")


@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    validate_readout_dataset(
        node.results["ds_raw"], node.parameters.use_state_discrimination
    )
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    figures = plot_per_qubit(
        plot_raw_data,
        node.results["ds_raw"],
        node.namespace["qubits"],
        figure_name="echo_lorentizan",
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.results["figures"] = figures
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")
    plt.show()


@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
