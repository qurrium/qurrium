"""ShadowUnveil - Utils (:mod:`qurry.qurrent.classical_shadow.utils`)"""

from typing import Optional
from collections.abc import Iterable, Hashable
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister

from .arguments import ShadowUnveilArguments
from ..randomized_measure.utils import bitstring_mapping_getter
from ...qurrium.experiment import After
from ...process.classical_shadow.unitary_set import U_M_GATES


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
            The selected qubits. Defaults to None.
        counts_used (Optional[Iterable[int]]):
            The index of the counts used. Defaults to None.
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
        - random_basis_with_clreg_index: The random basis with classical register index.
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
    assert isinstance(
        arguments.registers_mapping, dict
    ), f"registers_mapping {arguments.registers_mapping} is not dict."

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

    selected_qubits = [qi % arguments.actual_num_qubits for qi in selected_qubits]
    if len(set(selected_qubits)) != len(selected_qubits):
        raise ValueError(
            f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
        )

    random_basis_with_clreg_index = {
        n_u_i: {ci: single_basis[n_u_qi] for n_u_qi, ci in final_mapping.items()}
        for n_u_i, single_basis in arguments.random_basis.items()
    }

    selected_classical_registers = [final_mapping[qi] for qi in selected_qubits]

    return (
        counts,
        bitstring_mapping,
        arguments.registers_mapping,
        selected_qubits,
        selected_classical_registers,
        random_basis_with_clreg_index,
    )


def generate_random_basis(
    snapshots: int,
    unitary_located: list[int],
    random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
) -> dict[int, dict[int, int]]:
    """Generate the random basis for the classical shadow.

    Args:
        snapshots (int): The number of snapshots.
        unitary_located (list[int]): The list of selected qubits.
        random_unitary_seeds (Optional[dict[int, dict[int, int]]]):
            The random unitary seeds.
            This argument only takes input as type of `dict[int, dict[int, int]]`.
            The first key is the index for the random unitary operator.
            The second key is the index for the qubit.

            .. code-block:: python

                {
                    0: {0: 1234, 1: 5678},
                    1: {0: 2345, 1: 6789},
                    2: {0: 3456, 1: 7890},
                }

            If you want to generate the seeds for all random unitary operator,
            you can use the function :func:`generate_random_unitary_seeds`
            in :mod:`qurry.qurrium.utils.random_unitary`.

            .. code-block:: python

                from qurry import generate_random_unitary_seeds

                random_unitary_seeds = generate_random_unitary_seeds(100, 2)

    Returns:
        dict[int, dict[int, int]]: The random basis.
    """
    if any(not isinstance(qi, int) for qi in unitary_located):
        raise ValueError("All qubits in unitary_located should be integers.")

    random_basis_placeholder = np.random.randint(
        0, 3, size=(snapshots, len(unitary_located))
    ).tolist()
    random_basis = {
        n_u_i: {
            n_u_qi: (
                random_basis_placeholder[n_u_i][seed_i]
                if random_unitary_seeds is None
                else int(np.random.default_rng(random_unitary_seeds[n_u_i][seed_i]).integers(0, 3))
            )
            for seed_i, n_u_qi in enumerate(unitary_located)
        }
        for n_u_i in range(snapshots)
    }
    return random_basis


def check_random_basis(
    random_basis: dict[int, dict[int, int]],
    unitary_located: list[int],
) -> bool:
    """Check if the random basis is valid.

    Args:
        random_basis (dict[int, dict[int, int]]): The random basis.
        unitary_located (list[int]): The list of selected qubits.

    Returns:
        bool: True if the random basis is valid.

    Raise:
        ValueError: If the random basis is invalid.
    """
    if not isinstance(random_basis, dict):
        raise ValueError("random_basis should be a dictionary.")
    if not all(isinstance(qi, int) for qi in unitary_located):
        raise ValueError("All qubits in unitary_located should be integers.")

    invalid_dict = {}
    for k, v in random_basis.items():
        if not isinstance(k, int):
            invalid_dict[k] = f"Index '{k}' is not an integer, but '{type(k)}'."
        if not isinstance(v, dict):
            invalid_dict[k] = f"'{v}' is not a dictionary."
        if list(v) != unitary_located:
            invalid_dict[k] = (
                f"Keys '{list(v)}' do not match the expected qubits '{unitary_located}'."
            )
        if not all((isinstance(qi, int) and (0 <= q_basis < 3)) for qi, q_basis in v.items()):
            invalid_dict[k] = "All values should be integers in the range [0, 3) in the dictionary."

    if invalid_dict:
        raise ValueError(f"Invalid random_basis: {invalid_dict}")

    return True


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
        else f".{target_key}" + "" if len(old_name) < 1 else f".{old_name}"
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
