"""Simple dynamic-circuit active reset experiment.

The sequence measures the qubit and conditionally applies an x180 (pi) pulse
when the measured I quadrature indicates that the qubit is in |1>.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow both direct execution and `python -m dynamic_circuit_active_reset.active_reset`.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qm import SimulationConfig
from qm.qua import *

from qualang_tools.units import unit
from quam_config import Quam


DEFAULT_QUBIT = "q1"
DEFAULT_NUM_SHOTS = 1_000
DEFAULT_SIMULATION_DURATION = 20_000
DEFAULT_SIMULATION_TIMEOUT = 120


def plot_simulated_samples(samples):
    """Plot every simulated controller using the qm-qua controller plot API."""
    controller_names = list(samples.keys())
    if not controller_names:
        raise RuntimeError("The simulator returned no controller samples.")

    fig, axes = plt.subplots(
        nrows=len(controller_names),
        sharex=True,
        squeeze=False,
        figsize=(12, 4 * len(controller_names)),
    )
    for index, controller_name in enumerate(controller_names):
        plt.sca(axes[index, 0])
        samples[controller_name].plot()
        plt.title(controller_name)

    fig.suptitle("Dynamic Circuit Active Reset Simulation")
    fig.tight_layout()
    plt.show()


def build_active_reset_program(machine, qubit_name: str, num_shots: int):
    """Build a QUA program that resets |1> to |0> using measurement feedback."""
    u = unit(coerce_to_integer=True)
    qubit = machine.qubits[qubit_name]
    threshold = qubit.resonator.operations["readout"].threshold

    with program() as active_reset_program:
        shot = declare(int)
        initial_i = declare(fixed)
        initial_q = declare(fixed)
        final_i = declare(fixed)
        final_q = declare(fixed)

        initial_state = declare(int)
        final_state = declare(int)

        initial_state_stream = declare_stream()
        final_state_stream = declare_stream()
        shot_stream = declare_stream()

        with for_(shot, 0, shot < num_shots, shot + 1):
            save(shot, shot_stream)

            # Prepare |1> for half the shots and leave |0> unchanged for the rest.
            with if_(shot < num_shots // 2):
                qubit.xy.play("x180")

            align()
            qubit.resonator.measure("readout", qua_vars=(initial_i, initial_q))
            align()

            # Active reset: apply the pi gate only when measurement reports |1>.
            with if_(initial_i > threshold):
                assign(initial_state, 1)
                qubit.xy.play("x180")
            with else_():
                assign(initial_state, 0)

            qubit.resonator.wait(machine.depletion_time * u.ns)
            align()
            qubit.resonator.measure("readout", qua_vars=(final_i, final_q))

            with if_(final_i > threshold):
                assign(final_state, 1)
            with else_():
                assign(final_state, 0)

            save(initial_state, initial_state_stream)
            save(final_state, final_state_stream)
            qubit.resonator.wait(machine.depletion_time * u.ns)
            align()

        with stream_processing():
            shot_stream.save("shot")
            initial_state_stream.save_all("initial_state")
            final_state_stream.save_all("final_state")

    return active_reset_program


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit", default=DEFAULT_QUBIT)
    parser.add_argument("--num-shots", type=int, default=DEFAULT_NUM_SHOTS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute on hardware. Without this flag, simulate the program.",
    )
    parser.add_argument(
        "--simulation-duration",
        type=int,
        default=DEFAULT_SIMULATION_DURATION,
        help="Simulation duration in 4 ns clock cycles.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the simulation without opening a waveform plot.",
    )
    parser.add_argument(
        "--simulation-timeout",
        type=float,
        default=DEFAULT_SIMULATION_TIMEOUT,
        help="Seconds to wait for simulation completion before plotting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    machine = Quam.load()
    qua_program = build_active_reset_program(machine, args.qubit, args.num_shots)
    qmm = machine.connect()
    config = machine.generate_config()

    if args.execute:
        qm = qmm.open_qm(config)
        job = qm.execute(qua_program)
        print("Active reset is running on hardware.")
        print(job.execution_report())
    else:
        job = qmm.simulate(
            config,
            qua_program,
            SimulationConfig(duration=args.simulation_duration),
        )
        print("Simulation submitted successfully.")
        if not args.no_plot:
            print("Waiting for simulation to complete before plotting...")
            job.wait_until("Done", timeout=args.simulation_timeout)
            try:
                plot_simulated_samples(job.get_simulated_samples())
            except Exception as exc:
                print(
                    "Raw simulated samples could not be pulled; "
                    "plotting the waveform report instead."
                )
                print(f"Sample-pull error: {exc}")
                try:
                    job.plot_waveform_report_without_samples()
                except Exception as report_exc:
                    print("The waveform report could not be plotted either.")
                    print(f"Waveform-report error: {report_exc}")


if __name__ == "__main__":
    main()
