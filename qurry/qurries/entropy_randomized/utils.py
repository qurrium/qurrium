"""EntropyMeasureRandomized - Utility (:mod:`qurry.qurries.entropy_randomized.utils`)"""

import tqdm

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Operator

from .arguments import EMRArguments
from .tales import RandomizedMeasureTales
from ...qurrium import WCKeyable, naming_circuit
from ...process.randomized_measure import (
    generate_random_unitary,
    local_unitary_op_to_list,
    local_unitary_op_to_bloch_vector,
)
from ...tools import ParallelManager, set_pbar_description

DEFAULT_CLASSICAL_REGISTER_NAME = "m0"
"""The default name for classical register used for measurement."""


def make_samplied_circuit(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: WCKeyable,
    exp_name: str,
    registers_mapping: dict[int, int],
    single_unitary_dict: dict[int, Operator],
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
        single_unitary_dict (dict[int, Operator]):
            The dictionary of the unitary operator.

    Returns:
        QuantumCircuit: The circuit for the experiment.
    """

    qc_exp1 = target_circuit.copy(naming_circuit(target_circuit, target_key, f"{exp_name}_{idx}"))
    c_meas1 = ClassicalRegister(len(registers_mapping), DEFAULT_CLASSICAL_REGISTER_NAME)
    qc_exp1.add_register(c_meas1)

    qc_exp1.barrier()

    for qi, operator in single_unitary_dict.items():
        qc_exp1.append(operator.to_instruction(), [qi])

    for qi, ci in registers_mapping.items():
        qc_exp1.measure(qc_exp1.qubits[qi], c_meas1[ci])

    assert qc_exp1.cregs[-1] == c_meas1, (
        f"The last classical register should be the measurement register {c_meas1},"
        + f" but get {qc_exp1.cregs[-1]} in {qc_exp1.cregs}. From {exp_name} on index {idx}."
    )

    return qc_exp1


def make_unitary_op_pauli_coeff(
    idx: int, single_unitary_dict: dict[int, Operator]
) -> tuple[int, dict[int, list[list[complex]]], dict[int, tuple[float, float, float]]]:
    """Build the unitary operator and pauli coeff for the experiment.

    Args:
        idx (int):
            Index of the quantum circuit.
        single_unitary_dict (dict[int, Operator]):
            The dictionary of the unitary operator.

    Returns:
        A tuple containing the index, the dictionary of unitary operators,
        and the dictionary of pauli coefficients.
    """

    unitary_op = local_unitary_op_to_list(single_unitary_dict)

    return idx, unitary_op, local_unitary_op_to_bloch_vector(unitary_op)


def make_samplied_circuit_unitary_op_pauli_coeff(
    idx: int,
    target_circuit: QuantumCircuit,
    target_key: WCKeyable,
    exp_name: str,
    registers_mapping: dict[int, int],
    single_unitary_dict: dict[int, Operator],
) -> tuple[
    int,
    QuantumCircuit,
    dict[int, list[list[complex]]],
    dict[int, tuple[float, float, float]],
]:
    """Build the circuit, unitary operator and pauli coeff for the experiment.

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
        single_unitary_dict (dict[int, Operator]):
            The dictionary of the unitary operator.

    Returns:
        A tuple containing the index, the circuit for the experiment,
        the dictionary of unitary operators,
        and the dictionary of pauli coefficients.
    """

    idx, unitary_op, pauli_coeff = make_unitary_op_pauli_coeff(idx, single_unitary_dict)
    return (
        idx,
        make_samplied_circuit(
            idx,
            target_circuit,
            target_key,
            exp_name,
            registers_mapping,
            single_unitary_dict,
        ),
        unitary_op,
        pauli_coeff,
    )


def method_process(
    targets: list[tuple[WCKeyable, QuantumCircuit]],
    arguments: EMRArguments,
    pbar: tqdm.tqdm | None = None,
    multiprocess: bool = False,
) -> tuple[list[QuantumCircuit], RandomizedMeasureTales]:
    """The process method for building the circuits of the experiment.

    Args:
        targets (list[tuple[WCKeyable, QuantumCircuit]]):
            The circuits of the experiment.
        arguments (EMRArguments):
            The arguments of the experiment.
        pbar (tqdm.tqdm | None, optional):
            The progress bar for showing the progress of the experiment.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to False.

    Returns:
        A tuple containing a list of quantum circuits and a dictionary of additional information.
    """

    target_key, target_circuit = targets[0]
    target_key = "" if isinstance(target_key, int) else str(target_key)

    set_pbar_description(pbar, f"Preparing {arguments.times} random unitary.")
    unitary_dicts = generate_random_unitary(
        times=arguments.times,
        unitary_located=arguments.unitary_located,
        random_unitary_seeds=arguments.random_unitary_seeds,
    )

    set_pbar_description(pbar, f"Building {arguments.times} circuits.")
    if multiprocess:
        pool = ParallelManager()
        result_list = pool.starmap(
            make_samplied_circuit_unitary_op_pauli_coeff,
            [
                (
                    n_u_i,
                    target_circuit,
                    target_key,
                    arguments.exp_name,
                    arguments.registers_mapping,
                    unitary_dicts[n_u_i],
                )
                for n_u_i in range(arguments.times)
            ],
        )
    else:
        result_list = [
            make_samplied_circuit_unitary_op_pauli_coeff(
                n_u_i,
                target_circuit,
                target_key,
                arguments.exp_name,
                arguments.registers_mapping,
                unitary_dicts[n_u_i],
            )
            for n_u_i in range(arguments.times)
        ]

    assert [x[0] for x in result_list] == list(range(arguments.times)), (
        "The indices of the results are not correct."
        + f" Get {[x[0] for x in result_list]}, expect {list(range(arguments.times))}."
    )

    return [x[1] for x in result_list], RandomizedMeasureTales(
        {
            "unitary_operator": {i: u_op for i, _q, u_op, _p_c in result_list},
            "bloch_vector": {i: p_c for i, _q, _u_op, p_c in result_list},
        }
    )
