"""ZDirMagnetSquare - Utility (:mod:`qurry.qurries.magnet_square_z.utils`)"""

from qiskit import QuantumCircuit, ClassicalRegister


def circuit_method(
    target_circuit: QuantumCircuit, target_key: str, exp_name: str
) -> QuantumCircuit:
    """Build the circuit for the experiment.

    Args:
        target_circuit (QuantumCircuit): Target circuit.
        target_key (Hashable): Target key.
        exp_name (str): Experiment name.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """

    old_name = "" if isinstance(target_circuit.name, str) else target_circuit.name
    qc_exp1 = target_circuit.copy(
        f"{exp_name}_zdir"
        + ("" if target_key else f".{target_key}")
        + ("" if old_name else f".{old_name}")
    )
    c_meas1 = ClassicalRegister(qc_exp1.num_qubits, "c_m1")
    qc_exp1.add_register(c_meas1)
    qc_exp1.barrier()
    qc_exp1.measure(qc_exp1.qubits, c_meas1)

    return qc_exp1
