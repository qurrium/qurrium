"""String Operator - Utilities
(:mod:`qurry.qurries.string_operator.utils`)

"""

from typing import Union, Literal
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister


AvailableStringOperatorTypes = Union[Literal["i", "zy"], str]
"""Available string types.

- "i": Identity string operator.
- "zy": ZY string operator.
"""
AvailableStringOperatorUnits = Union[tuple[Literal["rx", "ry", "rz"], float], tuple[()]]
"""Available string operator units."""

STRING_OPERATOR_LIB: dict[
    AvailableStringOperatorTypes,
    dict[Union[int, Literal["filling"]], AvailableStringOperatorUnits],
] = {
    "i": {
        0: (),
        "filling": ("ry", -np.pi / 2),
        -1: (),
    },
    "zy": {
        0: ("rz", 0),
        1: ("rx", np.pi / 2),
        "filling": ("ry", -np.pi / 2),
        -2: ("rx", np.pi / 2),
        -1: ("rz", 0),
    },
}
"""Available string operator library.
"""

STRING_OPERATOR_LIB2 = {
    "i": {
        0: (),
        "filling": ("rx", np.pi / 2),
        -1: (),
    },
    "zy": {
        0: (),
        1: ("ry", -np.pi / 2),
        "filling": ("rx", np.pi / 2),
        -2: ("ry", -np.pi / 2),
        -1: (),
    },
}


def circuit_method(
    target_circuit: QuantumCircuit,
    target_key: str,
    exp_name: str,
    i: int,
    k: int,
    str_op: AvailableStringOperatorTypes,
) -> QuantumCircuit:
    """Build the circuit for the experiment.

    Args:
        target_circuit (QuantumCircuit):
            Target circuit.
        target_key (Hashable):
            Target key.
        exp_name (str):
            Experiment name.
        i (int):
            The index of beginning qubits in the quantum circuit.
        k (int):
            The index of ending qubits in the quantum circuit.
        str_op (AvailableStringOperatorTypes):
            The string operator.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """

    old_name = "" if isinstance(target_circuit.name, str) else target_circuit.name

    qc_exp1 = target_circuit.copy(
        f"{exp_name}_{i}to{k}_{str_op}"
        + ("" if target_key else f".{target_key}")
        + ("" if old_name else f".{old_name}")
    )
    c_meas1 = ClassicalRegister(2, "c_m1")
    qc_exp1.add_register(c_meas1)

    qc_exp1.barrier()

    bound_mapping = {
        (k + 1 + op if op < 0 else i + op): op
        for op in STRING_OPERATOR_LIB[str_op]
        if isinstance(op, int)
    }

    for ci, qi in enumerate(range(i, k + 1)):
        move = STRING_OPERATOR_LIB[str_op][
            (bound_mapping[qi] if qi in bound_mapping else "filling")
        ]
        if len(move) == 0:
            continue

        if move[0] == "rx":
            qc_exp1.rx(move[1], qi)
        elif move[0] == "ry":
            qc_exp1.ry(move[1], qi)
        qc_exp1.measure(qc_exp1.qubits[qi], c_meas1[ci])

    return qc_exp1
