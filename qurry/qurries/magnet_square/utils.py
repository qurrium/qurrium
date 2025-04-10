"""Magnetic Square - Utility
(:mod:`qurry.qurries.magnet_square.utils`)

"""

from qiskit import QuantumCircuit, ClassicalRegister


def circuit_method(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: str,
    exp_name: str,
    i: int,
    j: int,
) -> QuantumCircuit:
    """Build the circuit for the experiment.

    Args:
        idx (int):
            Index of the quantum circuit.
        target_circuit (QuantumCircuit):
            Target circuit.
        target_key (Hashable):
            Target key.
        exp_name (str):
            Experiment name.
        i (int):
            The index of the target qubit.
        j (int):
            The index of the target qubit.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """

    old_name = "" if isinstance(target_circuit.name, str) else target_circuit.name
    qc_exp1 = target_circuit.copy(
        f"{exp_name}_{idx}_{i}_{j}"
        + ("" if target_key else f".{target_key}")
        + ("" if old_name else f".{old_name}")
    )
    c_meas1 = ClassicalRegister(2, "c_m1")
    qc_exp1.add_register(c_meas1)

    qc_exp1.barrier()

    qc_exp1.measure(qc_exp1.qubits[i], c_meas1[0])
    qc_exp1.measure(qc_exp1.qubits[j], c_meas1[1])

    return qc_exp1
