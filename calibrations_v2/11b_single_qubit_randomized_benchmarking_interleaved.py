"""Class-based v2 migration for interleaved single-qubit randomized benchmarking."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

from calibration_utils.single_qubit_randomized_benchmarking import (
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from calibration_utils.single_qubit_randomized_benchmarking_interleaved import (
    Parameters,
    get_interleaved_gate_index,
)
from quam_config import Quam, create_machine

if __package__ in {None, ""}:
    from calibrations_v2.core import BaseCalibration, CalibrationOptions
else:
    from .core import BaseCalibration, CalibrationOptions


description = """
        SINGLE QUBIT RANDOMIZED BENCHMARKING - INTERLEAVED
The program consists in playing random sequences of Clifford gates and measuring the
state of the resonator afterward. Each random sequence is derived on the FPGA for the
maximum depth and played for each requested depth. Each truncated sequence ends with
the recovery gate found from a preloaded Cayley table.

In this version, a Clifford gate chosen by the user is interleaved between each random
gate in the sequence. This allows characterizing the fidelity of a specific gate.

State update:
    - The interleaved single-qubit gate fidelity: qubit.gate_fidelity[interleaved_gate_operation].
"""


class SingleQubitRandomizedBenchmarkingInterleaved(BaseCalibration[Parameters, Quam]):
    """Interleaved single-qubit randomized benchmarking using the v2 lifecycle."""

    def __init__(
        self,
        parameters: Parameters,
        machine: Quam | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="11b_single_qubit_randomized_benchmarking_interleaved",
            description=description,
            parameters=parameters,
            machine=machine,
            **kwargs,
        )

    def progress_total(self) -> int:
        return self.parameters.num_random_sequences

    def create_qua_program(self):
        """Create the sweep axes and generate the interleaved RB QUA program."""
        qubits = self.get_qubits()
        num_qubits = len(qubits)
        num_of_sequences = self.parameters.num_random_sequences
        strict_timing = self.parameters.use_strict_timing
        n_avg = self.parameters.num_shots
        max_circuit_depth = self.parameters.max_circuit_depth
        delta_clifford = self.parameters.delta_clifford
        if max_circuit_depth % delta_clifford != 0:
            raise ValueError("max_circuit_depth must be divisible by delta_clifford.")

        num_depths = max_circuit_depth // delta_clifford
        seed = self.parameters.seed
        interleaved_gate_index = get_interleaved_gate_index(
            self.parameters.interleaved_gate_operation
        )
        inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

        def generate_sequence(interleaved_gate_index):
            cayley = declare(int, value=c1_table.flatten().tolist())
            inv_list = declare(int, value=inv_gates)
            current_state = declare(int)
            step = declare(int)
            sequence = declare(int, size=2 * max_circuit_depth + 1)
            inv_gate = declare(int, size=2 * max_circuit_depth + 1)
            i = declare(int)
            rand = Random(seed=seed)

            assign(current_state, 0)
            with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
                assign(step, rand.rand_int(24))
                assign(current_state, cayley[current_state * 24 + step])
                assign(sequence[i], step)
                assign(inv_gate[i], inv_list[current_state])
                assign(step, interleaved_gate_index)
                assign(current_state, cayley[current_state * 24 + step])
                assign(sequence[i + 1], step)
                assign(inv_gate[i + 1], inv_list[current_state])

            return sequence, inv_gate

        def play_sequence(sequence_list, depth, qubit):
            i = declare(int)
            with for_(i, 0, i <= depth, i + 1):
                with switch_(sequence_list[i], unsafe=True):
                    with case_(0):
                        qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
                    with case_(1):
                        qubit.xy.play("x180")
                    with case_(2):
                        qubit.xy.play("y180")
                    with case_(3):
                        qubit.xy.play("y180")
                        qubit.xy.play("x180")
                    with case_(4):
                        qubit.xy.play("x90")
                        qubit.xy.play("y90")
                    with case_(5):
                        qubit.xy.play("x90")
                        qubit.xy.play("-y90")
                    with case_(6):
                        qubit.xy.play("-x90")
                        qubit.xy.play("y90")
                    with case_(7):
                        qubit.xy.play("-x90")
                        qubit.xy.play("-y90")
                    with case_(8):
                        qubit.xy.play("y90")
                        qubit.xy.play("x90")
                    with case_(9):
                        qubit.xy.play("y90")
                        qubit.xy.play("-x90")
                    with case_(10):
                        qubit.xy.play("-y90")
                        qubit.xy.play("x90")
                    with case_(11):
                        qubit.xy.play("-y90")
                        qubit.xy.play("-x90")
                    with case_(12):
                        qubit.xy.play("x90")
                    with case_(13):
                        qubit.xy.play("-x90")
                    with case_(14):
                        qubit.xy.play("y90")
                    with case_(15):
                        qubit.xy.play("-y90")
                    with case_(16):
                        qubit.xy.play("-x90")
                        qubit.xy.play("y90")
                        qubit.xy.play("x90")
                    with case_(17):
                        qubit.xy.play("-x90")
                        qubit.xy.play("-y90")
                        qubit.xy.play("x90")
                    with case_(18):
                        qubit.xy.play("x180")
                        qubit.xy.play("y90")
                    with case_(19):
                        qubit.xy.play("x180")
                        qubit.xy.play("-y90")
                    with case_(20):
                        qubit.xy.play("y180")
                        qubit.xy.play("x90")
                    with case_(21):
                        qubit.xy.play("y180")
                        qubit.xy.play("-x90")
                    with case_(22):
                        qubit.xy.play("x90")
                        qubit.xy.play("y90")
                        qubit.xy.play("x90")
                    with case_(23):
                        qubit.xy.play("-x90")
                        qubit.xy.play("y90")
                        qubit.xy.play("-x90")

        depths = np.arange(1, max_circuit_depth + 0.1, delta_clifford)
        self.namespace["sweep_axes"] = {
            "qubit": xr.DataArray(qubits.get_names()),
            "nb_of_sequences": xr.DataArray(
                np.arange(num_of_sequences),
                attrs={"long_name": "Number of sequences"},
            ),
            "depths": xr.DataArray(
                depths,
                attrs={"long_name": "Number of Clifford gates"},
            ),
        }

        with program() as qua_program:
            I, I_st, Q, Q_st, n, n_st = self.machine.declare_qua_variables()
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
            depth = declare(int)
            depth_target = declare(int)
            saved_gate = declare(int)
            m = declare(int)
            m_st = declare_stream()

            for multiplexed_qubits in qubits.batch():
                for qubit in multiplexed_qubits.values():
                    self.machine.initialize_qpu(target=qubit)
                align()

                with for_(m, 0, m < num_of_sequences, m + 1):
                    save(m, m_st)
                    sequence_list, inv_gate_list = generate_sequence(
                        interleaved_gate_index=interleaved_gate_index
                    )
                    assign(depth_target, 2)

                    with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1):
                        assign(saved_gate, sequence_list[depth])
                        assign(sequence_list[depth], inv_gate_list[depth - 1])
                        with if_(depth == depth_target):
                            with for_(n, 0, n < n_avg, n + 1):
                                for _, qubit in multiplexed_qubits.items():
                                    qubit.reset(
                                        self.parameters.reset_type,
                                        self.parameters.simulate,
                                    )
                                align()

                                for _, qubit in multiplexed_qubits.items():
                                    if strict_timing:
                                        with strict_timing_():
                                            play_sequence(sequence_list, depth, qubit)
                                    else:
                                        play_sequence(sequence_list, depth, qubit)
                                align()

                                for i, qubit in multiplexed_qubits.items():
                                    if self.parameters.use_state_discrimination:
                                        qubit.readout_state(state[i])
                                        save(state[i], state_st[i])
                                    else:
                                        qubit.resonator.measure(
                                            "readout",
                                            qua_vars=(I[i], Q[i]),
                                        )
                                        save(I[i], I_st[i])
                                        save(Q[i], Q_st[i])
                                align()
                            assign(depth_target, depth_target + 2 * delta_clifford)
                        assign(sequence_list[depth], saved_gate)

            with stream_processing():
                m_st.save("n")
                for i in range(num_qubits):
                    if self.parameters.use_state_discrimination:
                        state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                            num_depths
                        ).buffer(num_of_sequences).save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                            num_depths
                        ).buffer(num_of_sequences).save(f"I{i + 1}")
                        Q_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                            num_depths
                        ).buffer(num_of_sequences).save(f"Q{i + 1}")

        self.namespace["qua_program"] = qua_program
        return qua_program

    def analyse_data(self) -> None:
        """Process and fit the interleaved RB decay."""
        self.results["ds_raw"] = process_raw_dataset(self.results["ds_raw"], self)
        self.results["ds_fit"], fit_results = fit_raw_data(self.results["ds_raw"], self)
        self.results["fit_results"] = {
            key: asdict(value) for key, value in fit_results.items()
        }
        log_fitted_results(self.results["fit_results"], log_callable=self.log)
        self.outcomes = {
            qubit_name: ("successful" if fit_result["success"] else "failed")
            for qubit_name, fit_result in self.results["fit_results"].items()
        }

    def plot_data(self) -> None:
        """Plot the raw interleaved RB data with the fitted decay."""
        figure = plot_raw_data_with_fit(
            self.results["ds_raw"],
            self.namespace["qubits"],
            self.results["ds_fit"],
        )
        figure.suptitle(
            "Single qubit randomized benchmarking interleaved with "
            f"{self.parameters.interleaved_gate_operation}"
        )
        if self.options.plot_data:
            plt.show()
        self.results["figures"] = {"amplitude": figure}

    def update_state(self) -> None:
        """Update the in-memory gate fidelity for the interleaved operation."""
        with self.record_state_updates():
            for qubit in self.namespace["qubits"]:
                if self.outcomes[qubit.name] == "failed":
                    continue
                qubit.gate_fidelity[self.parameters.interleaved_gate_operation] = float(
                    1 - self.results["fit_results"][qubit.name]["error_per_gate"]
                )


if __name__ == "__main__":
    parameters = Parameters()
    parameters.use_state_discrimination = True
    parameters.reset_type = "active"
    parameters.interleaved_gate_operation = "x180"
    parameters.max_circuit_depth = 500
    parameters.delta_clifford = 5
    parameters.num_random_sequences = 10
    parameters.num_shots = 100
    parameters.use_strict_timing = True
    parameters.simulate = False

    calibration = SingleQubitRandomizedBenchmarkingInterleaved(
        parameters=parameters,
        options=CalibrationOptions(),
        machine=create_machine(qubit="q1"),
    )
    calibration.run()
