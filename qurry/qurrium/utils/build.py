"""Build Tools (:mod:`qurry.qurrium.utils.build`)"""

from qiskit import QuantumCircuit, ClassicalRegister

DEFAULT_COLLISION_PREFIX = "ori_"
"""The default prefix to add to the existed register name in case of collision."""


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


def is_cregs_name_collision(
    circuit: QuantumCircuit, new_creg_or_name: ClassicalRegister | str
) -> bool:
    """Whether the name of the new classical register collides with existing ones.

    Args:
        circuit (QuantumCircuit): The quantum circuit to check.
        new_creg_or_name (ClassicalRegister | str): The new classical register or its name.

    Raises:
        TypeError: If `new_creg_or_name` is neither a `ClassicalRegister` nor a `str`.

    Returns:
        bool: True if there is a name collision, False otherwise.
    """
    if isinstance(new_creg_or_name, ClassicalRegister):
        return circuit.has_register(new_creg_or_name)
    elif isinstance(new_creg_or_name, str):
        return new_creg_or_name in [reg.name for reg in circuit.cregs]
    else:
        raise TypeError(
            "The argument new_creg_or_name should be either a ClassicalRegister or a string."
        )


def rename_collision_cregs(
    circuit: QuantumCircuit,
    new_creg_or_name: ClassicalRegister | str,
    prefix: str = DEFAULT_COLLISION_PREFIX,
):
    """Raise ValueError if there is a name collision with existing classical registers.

    Args:
        circuit (QuantumCircuit): The quantum circuit to check.
        new_creg_or_name (ClassicalRegister | str): The new classical register or its name.
        prefix (str, optional): The prefix to add to the existed classical register name
            in case of collision. Defaults to DEFAULT_COLLISION_PREFIX.
    """
    if not is_cregs_name_collision(circuit, new_creg_or_name):
        return

    original_names = [reg.name for reg in circuit.cregs]
    for creg in circuit.cregs:
        creg._name = prefix + creg.name

    if is_cregs_name_collision(circuit, new_creg_or_name):
        name_of_collision = (
            new_creg_or_name.name
            if isinstance(new_creg_or_name, ClassicalRegister)
            else new_creg_or_name
        )
        renamed_names = [reg.name for reg in circuit.cregs]
        raise ValueError(
            f"The classical register name collision still exists after renaming with prefix '{prefix}'. "
            f"Original classical register names: {original_names}. "
            f"Renamed classical register names: {renamed_names}. "
            f"Current conflicting name: {name_of_collision}. "
            + "Please use another prefix to rename the existing classical registers, "
            + "or consider another name for the new classical register."
        )
