"""Build Tools (:mod:`qurry.qurrium.utils.build`)"""

from qiskit import QuantumCircuit


def decomposer(qc: QuantumCircuit, reps: int = 2) -> QuantumCircuit:
    """Decompose the circuit with giving times.

    Args:
        qc (QuantumCircuit): The circuit wanted to be decomposed.
        reps (int, optional): Decide the times of decomposing the circuit.
            Draw quantum circuit with composed circuit. Defaults to 2.

    Returns:
        QuantumCircuit: The decomposed circuit.
    """

    return qc.decompose(reps=reps)
