# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from utils.simulation import simulate_and_plot

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.fine_rabi import (
    Parameters,
    analyze_fine_rabi,
    log_analysis_results,
    operation_for_rotation,
    plot_fine_rabi,
    process_raw_dataset,
    pulses_per_repetition_group,
)
from quam_config import Quam, create_machine
from profiles import ProfileUpdater, load_profile
from utils.plotting_settings import plot_per_qubit

description = """
        FINE RABI CALIBRATION
Sweep drive amplitude while repeatedly applying complete gate groups that
ideally return the qubit to the ground state.

For rotation_type="PI", each group is (Xpi Xpi).
For rotation_type="PI_HALF", each group is (Xpi/2 Xpi/2 Xpi/2 Xpi/2).

Small amplitude errors accumulate with the number of groups, making the
optimal amplitude easier to identify from the flat return-to-ground response
and from the Fourier map along the repetition axis.

State update:
    - The x180 pulse amplitude, scaled by the fitted optimal amplitude factor.
"""


node = QualibrationNode[Parameters, Quam](
    name="04e_fine_rabi_calibration",
    description=description,
    parameters=Parameters(),
    machine=create_machine(),
)


node.machine = create_machine(qubit="q2")

node.machine.connect()  # Connect to the machine to fetch the qubits information and populate the node namespace if needed

node.machine.qmm.close_all_qms()


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debugging parameter overrides."""
    node.parameters.use_state_discrimination = True
    node.parameters.rotation_type = "PI"
    node.parameters.reset_type = "active"
    # node.parameters.amp_factor_spacing = "center_dense"

    # node.parameters.max_repetition_groups = 40
    # node.parameters.min_amp_factor = 0.8
    # node.parameters.max_amp_factor = 1.2
    # node.parameters.amp_factor_step = 0.01
    pass


def validate_readout_dataset(ds: xr.Dataset, use_state_discrimination: bool) -> None:
    """Ensure fetched results match the requested readout mode."""
    expected = {"state"} if use_state_discrimination else {"I", "Q"}
    unexpected = {"I", "Q"} if use_state_discrimination else {"state"}
    missing = expected - set(ds.data_vars)
    present_unexpected = unexpected & set(ds.data_vars)
    if missing or present_unexpected:
        raise RuntimeError(
            "Fine-Rabi readout mode mismatch: "
            f"use_state_discrimination={use_state_discrimination}, "
            f"dataset variables={sorted(ds.data_vars)}, "
            f"missing={sorted(missing)}, unexpected={sorted(present_unexpected)}"
        )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the fine-Rabi amplitude and repetition-group sweep."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation = operation_for_rotation(node.parameters.rotation_type)
    pulses_per_group = pulses_per_repetition_group(node.parameters.rotation_type)
    for qubit in qubits:
        if operation not in qubit.xy.operations:
            raise ValueError(f"{qubit.name} does not define operation {operation!r}.")

    amps = node.parameters.get_amp_factors()
    repetition_groups = node.parameters.get_repetition_groups()
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "repetition_group_count": xr.DataArray(
            repetition_groups,
            attrs={"long_name": "number of complete gate groups"},
        ),
        "amp_prefactor": xr.DataArray(
            amps,
            attrs={"long_name": "pulse amplitude prefactor"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        group_count = declare(int)
        group_index = declare(int)
        pulse_index = declare(int)
        a = declare(fixed)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(group_count, repetition_groups)):
                    with for_each_(a, amps.tolist()):
                        for _, qubit in multiplexed_qubits.items():
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                # log_callable=node.log,
                            )
                        align()

                        for _, qubit in multiplexed_qubits.items():
                            with for_(
                                group_index,
                                0,
                                group_index < group_count,
                                group_index + 1,
                            ):
                                with for_(
                                    pulse_index,
                                    0,
                                    pulse_index < pulses_per_group,
                                    pulse_index + 1,
                                ):
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
                    state_st[i].buffer(len(amps)).buffer(
                        len(repetition_groups)
                    ).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amps)).buffer(
                        len(repetition_groups)
                    ).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(
                        len(repetition_groups)
                    ).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, figure, waveform_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": figure,
        "waveform_report": waveform_report,
        "samples": samples,
    }
    plt.show()


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
    validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Save_raw_results}
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    validate_readout_dataset(
        node.results["ds_raw"], node.parameters.use_state_discrimination
    )
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = analyze_fine_rabi(
        node.results["ds_raw"], node
    )
    node.results["fit_results"] = fit_results
    log_analysis_results(fit_results, log_callable=node.log)
    node.outcomes = {qubit.name: "successful" for qubit in node.namespace["qubits"]}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    figures = plot_per_qubit(
        plot_fine_rabi,
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.parameters.use_state_discrimination,
        node.parameters.rotation_type,
        fits=node.results["ds_fit"],
        figure_name="fine_rabi",
    )
    plt.show()
    node.results["figures"] = figures
    if "calibration_run_directory" in node.namespace:
        figures_directory = CalibrationSaver().save_figures(
            node.namespace["calibration_run_directory"],
            node.results["figures"],
        )
        node.log(f"Calibration figures saved to {figures_directory}")


# %% {Propose_profile_update}
@node.run_action(skip_if=node.parameters.simulate)
def propose_profile_update(node: QualibrationNode[Parameters, Quam]):
    """Stage x180 amplitude updates from the fitted Fine Rabi amplitude factor."""
    updates = {}
    profile_name = current_profile_name()
    profile = load_profile(profile_name)
    qubit_profiles = profile["qubits"]["qubits"]
    pulse_profiles = profile["pulses"]["pulses"]

    for q in node.namespace["qubits"]:
        result = node.results.get("fit_results", {}).get(q.name)
        if not result:
            node.log(
                f"Profile update skipped for {q.name}: no Fine Rabi optimum was found."
            )
            continue

        opt_amp_factor = float(result["optimal_amp_prefactor"])
        if not np.isfinite(opt_amp_factor) or opt_amp_factor <= 0:
            node.log(
                f"Profile update skipped for {q.name}: invalid Fine Rabi amplitude factor "
                f"{opt_amp_factor!r}."
            )
            continue

        qubit_profile = qubit_profiles[q.name]
        if "x180" not in qubit_profile["operations"]:
            node.log(
                f"Profile update skipped for {q.name}: profile has no x180 operation."
            )
            continue

        pulse_name = qubit_profile["operations"]["x180"]
        current_amplitude = float(pulse_profiles[q.name][pulse_name]["amplitude"])
        updates[f"pulses.json.pulses.{q.name}.{pulse_name}.amplitude"] = (
            current_amplitude * opt_amp_factor
        )

    if updates:
        proposal = ProfileUpdater().stage(node.name, updates, profile_name=profile_name)
        ProfileUpdater().confirm_and_apply(proposal)
