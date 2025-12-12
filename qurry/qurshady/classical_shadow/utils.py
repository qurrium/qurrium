"""ShadowUnveil - Utils (:mod:`qurry.qurrent.classical_shadow.utils`)"""

from qiskit import QuantumCircuit, ClassicalRegister

from ...qurrium import WCKeyable
from ...process.classical_shadow import ShadowRandomBasis


def make_samplied_circuit(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: WCKeyable,
    exp_name: str,
    registers_mapping: dict[int, int],
    single_random_basis: dict[int, int],
    shadow_basis: ShadowRandomBasis,
) -> QuantumCircuit:
    """Build the circuit for the experiment.

    Args:
        idx (int):
            Index of the quantum circuit.
        target_circuit (QuantumCircuit):
            Target circuit.
        target_key (WCKeyable):
            Target key.
        exp_name (str):
            Experiment name.
        registers_mapping (dict[int, int]):
            The mapping of the index of selected qubits to the index of the classical register.
        single_random_basis (dict[int, int]):
            The single random basis for each qubit.
        shadow_basis (ShadowRandomBasis):
            The shadow random basis.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """
    if not isinstance(shadow_basis, ShadowRandomBasis):
        raise TypeError(
            f"The shadow_basis should be an instance of ShadowRandomBasis, but get {type(shadow_basis)}"
        )

    old_name = "" if isinstance(target_circuit.name, str) else target_circuit.name

    qc_exp1 = target_circuit.copy(
        f"{exp_name}_{idx}" + ""
        if len(str(target_key)) < 1
        else f".{target_key}" + ""
        if len(old_name) < 1
        else f".{old_name}"
    )
    c_meas1 = ClassicalRegister(
        len(registers_mapping),
        None if "m1" in [reg.name for reg in (qc_exp1.qregs + qc_exp1.cregs)] else "m1",
    )
    qc_exp1.add_register(c_meas1)

    qc_exp1.barrier()

    for qi, um in single_random_basis.items():
        for g in shadow_basis.gates_tuple[um]:
            qc_exp1.append(g.copy(), [qi])

    for qi, ci in registers_mapping.items():
        qc_exp1.measure(qc_exp1.qubits[qi], c_meas1[ci])

    assert qc_exp1.cregs[-1] == c_meas1, (
        f"The last classical register should be the measurement register {c_meas1},"
        + f" but get {qc_exp1.cregs[-1]} in {qc_exp1.cregs}. From {exp_name} on index {idx}."
    )

    return qc_exp1
