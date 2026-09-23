"""Class-based calibration for 04e_fine_rabi_calibration."""

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
from utils.qm_session import qm_session
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
from utils.plotting_settings import plot_per_qubit

if __package__ in {None, ""}:
    from calibrations.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions

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
# %% {Simulate}
# %% {Execute}
# %% {Load_historical_data}
# %% {Save_raw_results}
# %% {Analyse_data}
# %% {Plot_data}
# %% {Propose_profile_update}


class FineRabiCalibration(BaseCalibration[Parameters, Quam]):
    """Class-based calibration for ``calibrations/04e_fine_rabi_calibration.py``."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="04e_fine_rabi_calibration",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def create_qua_program(self):
        node = self
        """Create the fine-Rabi amplitude and repetition-group sweep."""
        node.namespace["qubits"] = qubits = self.get_qubits()
        num_qubits = len(qubits)

        operation = operation_for_rotation(node.parameters.rotation_type)
        pulses_per_group = pulses_per_repetition_group(node.parameters.rotation_type)
        for qubit in qubits:
            if operation not in qubit.xy.operations:
                raise ValueError(
                    f"{qubit.name} does not define operation {operation!r}."
                )

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

        self.configure_acquisition()

        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            group_count = declare(int)
            group_index = declare(int)
            pulse_index = declare(int)
            a = declare(fixed)
            if node.parameters.use_state_discrimination:
                state = [declare(int) for _ in range(num_qubits)]
                state_st = [self.declare_state_stream() for _ in range(num_qubits)]

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()

                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(group_count, repetition_groups)):
                        with for_each_(a, amps.tolist()):
                            for _, qubit in multiplexed_qubits.items():
                                self.reset_qubit(qubit)
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
                                    self.readout_state(qubit, state[i])
                                    self.save_readout_state(state[i], state_st[i])
                                else:
                                    self.measure_readout(qubit, qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    self.process_readout_streams(
                        i, state_st if self.parameters.use_state_discrimination else None,
                        I_st, Q_st,
                    )

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
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

    def execute_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])
            dataset = self.fetch_result_dataset(job)
        validate_readout_dataset(dataset, node.parameters.use_state_discrimination)
        node.results["ds_raw"] = self.annotate_readout_dataset(dataset)

    def load_data(self):
        node = self
        load_data_id = node.parameters.load_data_id
        node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = self.get_qubits()

    def save_raw_results(self):
        node = self
        output_directory = CalibrationSaver().save_xarray(
            node.name,
            node.results["ds_raw"],
            profile_name=current_profile_name(),
            parameters=node.parameters,
        )
        node.namespace["calibration_run_directory"] = output_directory
        node.log(f"Raw calibration results saved to {output_directory}")

    def analyse_data(self):
        self.prepare_acquisition_results()
        node = self
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

    def plot_data(self):
        node = self
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

    def profile_updates(self):
        return self.pulse_profile_updates(
            "x180", amplitude_scale="optimal_amp_prefactor",
        )


if __name__ == "__main__":
    parameters = Parameters()
    parameters.acquisition = "averaged"  # or "single_shot" to retain every measurement

    parameters.use_state_discrimination = False
    parameters.rotation_type = "PI"
    parameters.reset_type = "thermal"
    parameters.num_shots = 100
    parameters.max_repetition_groups = 200
    parameters.amp_factor_step = 0.0021

    options = CalibrationOptions()

    calibration = FineRabiCalibration(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
