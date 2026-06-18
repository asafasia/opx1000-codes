"""Minimal single-qubit Qiskit circuit example."""

from qiskit import QuantumCircuit

from gate_level.runner import Runner, result_counts

QUBIT = "q9"
SHOTS = 1000


def build_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    return circuit


def circuit_text(circuit: QuantumCircuit) -> str:
    lines = []
    for instruction in circuit.data:
        operation = instruction.operation
        qargs = ", ".join(
            f"q[{circuit.find_bit(qubit).index}]" for qubit in instruction.qubits
        )
        cargs = ", ".join(
            f"c[{circuit.find_bit(bit).index}]" for bit in instruction.clbits
        )
        target = f" -> {cargs}" if cargs else ""
        lines.append(f"{operation.name} {qargs}{target}".strip())
    return "\n".join(lines)


def main() -> None:
    circuit = build_circuit()
    runner = Runner(qubit=QUBIT)
    transpiled = runner.transpile(circuit)

    print("Input circuit:")
    print(circuit_text(circuit))
    print("\nTranspiled circuit:")
    print(circuit_text(transpiled))

    result = runner.submit(transpiled, shots=SHOTS, do_transpile=False)
    counts = result_counts(result)

    print("\nResult:")
    print(dict(counts) if counts is not None else result)


if __name__ == "__main__":
    main()
