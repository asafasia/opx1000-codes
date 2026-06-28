"""Readout integration-weight optimization from sliced demodulated traces."""

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
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from calibration_io import CalibrationSaver, current_profile_name
from calibration_utils.readout_weights_optimization import (
    Parameters,
    kernel_to_segments,
    plot_readout_weight_traces,
    process_sliced_traces,
    save_kernel_artifacts,
)
from profiles import ProfileUpdater
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from quam_config import Quam, create_machine
from utils.simulation import simulate_and_plot

if __package__ in {None, ""}:
    from calibrations_v2.base import BaseCalibration, CalibrationOptions
else:
    from .base import BaseCalibration, CalibrationOptions


description = """
        READOUT WEIGHTS OPTIMIZATION
Acquire averaged sliced-demod traces for |g> and |e>, calculate the state-difference
trace, normalize it into an optimal complex kernel, and save the traces plus profile
kernel under profiles/<active-profile>/kernels/.

State update:
    - pulses.json.pulses.<qubit>.readout.integration_weights
"""


class ReadoutWeightsOptimization(BaseCalibration[Parameters, Quam]):
    """Optimize the readout integration kernel from sliced demodulation."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="10d_readout_weights_optimization",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    @property
    def slice_length_ns(self) -> int:
        return int(4 * self.parameters.division_length_clock_cycles)

    def _configure_integration_weight_mode(self, qubits, operation_name: str) -> None:
        if self.parameters.use_current_integration_weights:
            self.log("Using current profile integration weights for sliced readout.")
            return

        for qubit in qubits:
            operation = qubit.resonator.operations[operation_name]
            operation.integration_weights = [[1.0, int(operation.length)]]
            operation.integration_weights_angle = 0.0
        self.log("Using flat integration weights with zero angle for sliced readout.")

    def create_qua_program(self):
        node = self
        u = unit(coerce_to_integer=True)
        node.namespace["qubits"] = qubits = get_qubits(node)
        num_qubits = len(qubits)
        operation_name = node.parameters.operation
        division_length = int(node.parameters.division_length_clock_cycles)
        slice_length_ns = 4 * division_length
        n_avg = int(node.parameters.num_shots)

        if division_length <= 0:
            raise ValueError("division_length_clock_cycles must be positive.")
        if n_avg <= 0:
            raise ValueError("num_shots must be positive.")

        readout_lengths = []
        for qubit in qubits:
            if operation_name not in qubit.resonator.operations:
                raise ValueError(
                    f"{qubit.name} has no resonator operation {operation_name!r}."
                )
            readout_length = int(qubit.resonator.operations[operation_name].length)
            if readout_length % slice_length_ns != 0:
                raise ValueError(
                    f"{qubit.name} {operation_name!r} length {readout_length} ns is not "
                    f"divisible by the sliced-demod chunk {slice_length_ns} ns."
                )
            readout_lengths.append(readout_length)
        if len(set(readout_lengths)) != 1:
            raise ValueError("All selected qubits must use the same readout length.")

        node._configure_integration_weight_mode(qubits, operation_name)

        number_of_divisions = readout_lengths[0] // slice_length_ns
        node.namespace["number_of_divisions"] = number_of_divisions
        node.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "time_slice": xr.DataArray(
                np.arange(1, number_of_divisions + 1),
                attrs={"long_name": "sliced-demod index"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            n = declare(int)
            ind = declare(int)
            n_st = declare_stream()

            IIg = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            IQg = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            QIg = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            QQg = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            IIe = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            IQe = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            QIe = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
            QQe = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]

            streams = {
                name: [declare_stream() for _ in range(num_qubits)]
                for name in ("IIg", "IQg", "QIg", "QQg", "IIe", "IQe", "QIe", "QQe")
            }

            for multiplexed_qubits in qubits.batch():
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)

                    for qubit in multiplexed_qubits.values():
                        # qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        pass
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure_sliced(
                            operation_name,
                            segment_length=division_length,
                            qua_vars=(IIg[i], IQg[i], QIg[i], QQg[i]),
                        )
                        with for_(ind, 0, ind < number_of_divisions, ind + 1):
                            save(IIg[i][ind], streams["IIg"][i])
                            save(IQg[i][ind], streams["IQg"][i])
                            save(QIg[i][ind], streams["QIg"][i])
                            save(QQg[i][ind], streams["QQg"][i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    align()

                    for qubit in multiplexed_qubits.values():
                        # qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        pass
                    align()
                    for qubit in multiplexed_qubits.values():
                        qubit.xy.play("x180")
                    align()
                    for qubit in multiplexed_qubits.values():
                        qubit.resonator.wait(
                            node.parameters.xy_to_readout_delay_in_ns * u.ns
                        )

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure_sliced(
                            operation_name,
                            segment_length=division_length,
                            qua_vars=(IIe[i], IQe[i], QIe[i], QQe[i]),
                        )
                        with for_(ind, 0, ind < number_of_divisions, ind + 1):
                            save(IIe[i][ind], streams["IIe"][i])
                            save(IQe[i][ind], streams["IQe"][i])
                            save(QIe[i][ind], streams["QIe"][i])
                            save(QQe[i][ind], streams["QQe"][i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    align()

            with stream_processing():
                n_st.save("n")
                for i in range(num_qubits):
                    for name, stream_list in streams.items():
                        stream_list[i].buffer(number_of_divisions).average().save(
                            f"{name}{i + 1}"
                        )

        return node.namespace.get("qua_program")

    def simulate_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        samples, fig, wf_report = simulate_and_plot(
            qmm, config, node.namespace["qua_program"], node.parameters
        )
        node.results["simulation"] = {
            "figure": fig,
            "wf_report": wf_report,
            "samples": samples,
        }

    def execute_qua_program(self):
        node = self
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            dataset = None
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            node.log(job.execution_report())
        node.results["ds_raw"] = dataset

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

    def load_data(self):
        node = self
        load_data_id = node.parameters.load_data_id
        node.load_from_id(node.parameters.load_data_id)
        node.parameters.load_data_id = load_data_id
        node.namespace["qubits"] = get_qubits(node)

    def analyse_data(self):
        node = self
        analysed = process_sliced_traces(
            node.results["ds_raw"],
            slice_length_ns=node.slice_length_ns,
        )
        node.results["ds_fit"] = analysed
        output_directory = save_kernel_artifacts(
            profile_name=current_profile_name(),
            experiment_name=node.name,
            analysed=analysed,
            parameters=node.parameters,
        )
        node.namespace["kernel_artifact_directory"] = output_directory
        node.log(f"Readout traces and kernels saved to {output_directory}")
        node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}

    def plot_data(self):
        node = self
        figures = {
            "readout_weight_traces": plot_readout_weight_traces(
                node.results["ds_fit"],
                node.namespace["qubits"],
            )
        }
        plt.show()
        node.results["figures"] = figures
        if "calibration_run_directory" in node.namespace:
            figures_directory = CalibrationSaver().save_figures(
                node.namespace["calibration_run_directory"],
                node.results["figures"],
            )
            node.log(f"Calibration figures saved to {figures_directory}")

    def update_state(self):
        node = self
        with node.record_state_updates():
            for q in node.namespace["qubits"]:
                weights = node.results["ds_fit"].profile_kernel.sel(qubit=q.name).values
                q.resonator.operations[
                    node.parameters.operation
                ].integration_weights = kernel_to_segments(
                    weights,
                    node.slice_length_ns,
                )

    def propose_profile_update(self):
        node = self
        if node.parameters.operation != "readout":
            node.log(
                f"Profile update skipped: operation {node.parameters.operation!r} "
                "does not use the profile's default readout pulse."
            )
            return

        updates = {}
        for q in node.namespace["qubits"]:
            weights = node.results["ds_fit"].profile_kernel.sel(qubit=q.name).values
            updates[f"pulses.json.pulses.{q.name}.readout.integration_weights"] = (
                kernel_to_segments(
                    weights,
                    node.slice_length_ns,
                )
            )

        if updates:
            proposal = ProfileUpdater().stage(
                node.name,
                updates,
                profile_name=current_profile_name(),
            )
            ProfileUpdater().confirm_and_apply(proposal)


if __name__ == "__main__":
    parameters = Parameters()
    parameters.num_shots = 100000
    parameters.division_length_clock_cycles = 40
    parameters.use_current_integration_weights = False
    parameters.reset_type = "active"

    options = CalibrationOptions()

    calibration = ReadoutWeightsOptimization(
        parameters=parameters,
        options=options,
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
