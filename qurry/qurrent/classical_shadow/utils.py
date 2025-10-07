"""ShadowUnveil - Utils (:mod:`qurry.qurrent.classical_shadow.utils`)"""

from typing import Optional
from collections.abc import Iterable, Hashable

from qiskit import QuantumCircuit, ClassicalRegister

from .arguments import ShadowUnveilArguments
from ...qurrium.utils import bitstring_mapping_getter
from ...qurrium.experiment import After
from ...process.utils import counts_list_recount_pyrust
from ...process.classical_shadow.rho_process.unitary_set import U_M_GATES


def inner_process_analyze(
    selected_qubits: Optional[Iterable[int]],
    counts_used: Optional[Iterable[int]],
    arguments: ShadowUnveilArguments,
    afterwards: After,
):
    """The inner process for
    :meth:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment.analyze`
    in :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`.

    Args:
        selected_qubits (Optional[Iterable[int]]):
            The selected qubits.
        counts_used (Optional[Iterable[int]]):
            The index of the counts used.
        arguments (ShadowUnveilArguments):
            The arguments of
            :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`.
        afterwards (After)
            The afterwards of
            :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`.

    Return:
        - counts: The counts of the measurements after slice range.
        - bitstring_mapping: The mapping from bitstrings to qubits.
        - register_mapping: The mapping from qubits to classical registers.
        - selected_qubits: The selected qubits.
        - selected_classical_registers: The selected classical registers.
        - random_basis_array: The random basis array.
    """

    if selected_qubits is None:
        raise ValueError("selected_qubits should be specified.")
    assert arguments.unitary_located is not None, "unitary_located should be specified."
    assert arguments.random_basis is not None, "random_basis should be given here."

    if len(arguments.random_basis) != arguments.snapshots:
        raise ValueError(
            f"The number of random basis should be {arguments.snapshots}, "
            + f"but got {len(arguments.random_basis)}."
        )
    assert isinstance(arguments.registers_mapping, dict), (
        f"registers_mapping {arguments.registers_mapping} is not dict."
    )

    if isinstance(counts_used, Iterable):
        if max(counts_used) >= len(afterwards.counts):
            raise ValueError(
                "counts_used should be less than "
                f"{len(afterwards.counts)}, but get {max(counts_used)}."
            )
        counts = [afterwards.counts[i] for i in counts_used]
    elif counts_used is not None:
        raise ValueError(f"counts_used should be Iterable, but get {type(counts_used)}.")
    else:
        counts = afterwards.counts

    bitstring_mapping, final_mapping = bitstring_mapping_getter(counts, arguments.registers_mapping)

    # Remove multiple classical registers clusters, leave only one cluster by Qurrium
    counts = counts_list_recount_pyrust(
        counts, len(next(iter(counts[0].keys()))), list(final_mapping.values())
    )

    selected_qubits = [qi % arguments.actual_num_qubits for qi in selected_qubits]
    if len(set(selected_qubits)) != len(selected_qubits):
        raise ValueError(
            f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
        )
    selected_clregs_sorted = sorted([arguments.registers_mapping[qi] for qi in selected_qubits])
    all_clregs = sorted(arguments.registers_mapping.values())

    random_basis_array = []
    for i in range(len(arguments.random_basis)):
        tmp = {
            ci: arguments.random_basis[i][n_u_qi]
            for n_u_qi, ci in arguments.registers_mapping.items()
        }
        random_basis_array.append([tmp[j] for j in all_clregs])

    return (
        counts,
        bitstring_mapping,
        arguments.registers_mapping,
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
