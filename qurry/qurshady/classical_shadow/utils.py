"""ShadowUnveil - Utils (:mod:`qurry.qurrent.classical_shadow.utils`)"""

from typing import Optional
from collections.abc import Iterable, Hashable

from qiskit import QuantumCircuit, ClassicalRegister

from ...qurrium.utils import bitstring_mapping_getter
from ...process.utils import counts_list_recount_pyrust
from ...process.classical_shadow.rho_process.unitary_set import U_M_GATES


def inner_process_analyze(
    selected_qubits: Iterable[int],
    registers_mapping: dict[int, int],
    snapshots: int,
    num_qubits: int,
    random_basis: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
    counts_used: Optional[Iterable[int]],
):
    """The inner process for
    :meth:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment.analyze`
    in :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`.

    Args:
        selected_qubits (Iterable[int]):
            The selected qubits.
        registers_mapping (Optional[dict[int, int]]):
            The mapping of the classical registers with quantum registers.
        snapshots (Optional[int]):
            The number of random basis for classical shadow.
        num_qubits (Optional[int]):
            The number of qubits.
        random_basis (Optional[dict[int, dict[int, int]]]):
            The random basis for classical shadow.

        counts (list[dict[str, int]]):
            The counts of the experiment.
        counts_used (Optional[Iterable[int]]):
            The selected counts used for analysis.

    Return:
        - counts: The counts of the measurements after slice range.
        - bitstring_mapping: The mapping from bitstrings to qubits.
        - register_mapping: The mapping from qubits to classical registers.
        - selected_qubits: The selected qubits.
        - selected_classical_registers: The selected classical registers.
        - random_basis_array: The random basis array.
    """

    if len(random_basis) != snapshots:
        raise ValueError(
            f"The number of random basis should be {snapshots}, " + f"but got {len(random_basis)}."
        )
    if not isinstance(registers_mapping, dict):
        raise ValueError(
            "The registers_mapping should be dict, " + f"but got {type(registers_mapping)}."
        )
    if isinstance(counts_used, Iterable):
        if max(counts_used) >= len(counts):
            raise ValueError(
                f"counts_used should be less than {len(counts)}, but get {max(counts_used)}."
            )
        counts = [counts[i] for i in counts_used]

    bitstring_mapping, final_mapping = bitstring_mapping_getter(counts, registers_mapping)

    # Remove multiple classical registers clusters, leave only one cluster by Qurrium
    counts = counts_list_recount_pyrust(
        counts, len(next(iter(counts[0].keys()))), list(final_mapping.values())
    )

    selected_qubits = [qi % num_qubits for qi in selected_qubits]
    if len(set(selected_qubits)) != len(selected_qubits):
        raise ValueError(
            f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
        )
    selected_clregs_sorted = sorted([registers_mapping[qi] for qi in selected_qubits])
    all_clregs = sorted(registers_mapping.values())

    random_basis_array = []
    for i in range(len(random_basis)):
        tmp = {ci: random_basis[i][n_u_qi] for n_u_qi, ci in registers_mapping.items()}
        random_basis_array.append([tmp[j] for j in all_clregs])

    return (
        counts,
        bitstring_mapping,
        registers_mapping,
        selected_qubits,
        selected_clregs_sorted,
        random_basis_array,
    )


def circuit_method_core(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: Hashable,
    exp_name: str,
    registers_mapping: dict[int, int],
    single_unitary_um: dict[int, int],
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
        registers_mapping (dict[int, int]):
            The mapping of the index of selected qubits to the index of the classical register.
        single_unitary_dict (dict[int, Operator]):
            The dictionary of the unitary operator.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """

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

    for qi, um in single_unitary_um.items():
        qc_exp1.append(U_M_GATES[um], [qi])

    for qi, ci in registers_mapping.items():
        qc_exp1.measure(qc_exp1.qubits[qi], c_meas1[ci])

    assert qc_exp1.cregs[-1] == c_meas1, (
        f"The last classical register should be the measurement register {c_meas1},"
        + f" but get {qc_exp1.cregs[-1]} in {qc_exp1.cregs}. From {exp_name} on index {idx}."
    )

    return qc_exp1
