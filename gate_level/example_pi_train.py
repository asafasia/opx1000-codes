"""Pi-train example using the gate-level Runner."""

from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from gate_level.runner import Runner, result_counts_list

QUBIT = "q1"
SHOTS = 1000
PI_COUNTS = list(range(0, 51))
OUTPUT_FIGURE = Path(__file__).with_name("pi_train_population.png")


def build_pi_train_circuit(num_pi_gates: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    for _ in range(num_pi_gates):
        circuit.x(0)
    circuit.measure(0, 0)
    circuit.metadata = {"num_pi_gates": num_pi_gates}
    return circuit


def excited_population(counts: dict[str, int]) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return float("nan")
    return counts.get("1", 0) / shots


def plot_population(pi_counts: list[int], populations: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(pi_counts, populations, marker="o", color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Number of pi gates")
    ax.set_ylabel("Excited-state population")
    ax.set_title(f"Pi train on {QUBIT}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=160)
    plt.close(fig)


def main() -> None:
    circuits = [build_pi_train_circuit(n) for n in PI_COUNTS]
    runner = Runner(qubit=QUBIT)
    # These circuits already use backend-native x/measure operations, so skip
    # Qiskit's transpiler and submit the full list in one backend run.
    result = runner.submit(circuits, shots=SHOTS, do_transpile=False)

    counts_by_circuit = result_counts_list(result)
    populations = [excited_population(dict(counts)) for counts in counts_by_circuit]
    plot_population(PI_COUNTS, populations)

    for num_pi_gates, counts, population in zip(
        PI_COUNTS, counts_by_circuit, populations
    ):
        print(f"n={num_pi_gates:2d} population={population:.4f} counts={dict(counts)}")
    print(f"\nSaved plot: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
