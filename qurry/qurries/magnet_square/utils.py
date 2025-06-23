"""Magnetic Square - Utility (:mod:`qurry.qurries.magnet_square.utils`)"""

from typing import Union, Literal

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator


def circuit_method(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: str,
    exp_name: str,
    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]],
    i: int,
    j: int,
) -> QuantumCircuit:
    """Build the circuit for the experiment.

    Args:
        idx (int): Index of the quantum circuit.
        target_circuit (QuantumCircuit): Target circuit.
        target_key (Hashable): Target key.
        exp_name (str): Experiment name.
        unitary_operator (Union[Operator, Gate, Literal["x", "y", "z"]]):
            The unitary operator to apply.
            It can be a `qiskit.quantum_info.Operator`, a `qiskit.circuit.Gate`, or a string
            representing the axis of rotation ('x', 'y', or 'z').
        i (int): The index of the target qubit.
        j (int): The index of the target qubit.

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

    if isinstance(unitary_operator, Operator):
        qc_exp1.unitary(unitary_operator, [qc_exp1.qubits[i]], label="U")
        qc_exp1.unitary(unitary_operator, [qc_exp1.qubits[j]], label="U")
    elif isinstance(unitary_operator, Gate):
        qc_exp1.append(unitary_operator, [qc_exp1.qubits[i]])
        qc_exp1.append(unitary_operator, [qc_exp1.qubits[j]])
    elif unitary_operator == "x":
        qc_exp1.x(qc_exp1.qubits[i])
        qc_exp1.x(qc_exp1.qubits[j])
    elif unitary_operator == "y":
        qc_exp1.y(qc_exp1.qubits[i])
        qc_exp1.y(qc_exp1.qubits[j])
    elif unitary_operator == "z":
        ...
    else:
        raise ValueError(
            f"Invalid unitary operator: {unitary_operator}. "
            "It should be an Operator, Gate, or one of 'x', 'y', 'z'."
        )

    qc_exp1.measure(qc_exp1.qubits[i], c_meas1[0])
    qc_exp1.measure(qc_exp1.qubits[j], c_meas1[1])

    return qc_exp1
