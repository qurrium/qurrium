"""EchoListenRandomized - Utility (:mod:`qurry.qurries.echo_randomized.utils`)"""

from typing import Union, Optional, Literal
import warnings
import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.transpiler.passmanager import PassManager

from .arguments import ELRArguments
from .exceptions import (
    OverlapArgumentsUnfulfilled,
    MSG_OVERLAPPING_GIVEN,
    OverlapComparisonSizeDifferent,
    NSG_OVERLAPPING_SIZE,
)
from ...qurrium import WCKeyable, TranspileArgs
from ...qurrium.exceptions import TranspileConfigurationIgnored
from ..entropy_randomized import EntropyMeasureTalesTypes
from ..entropy_randomized.utils import make_samplied_circuit, make_unitary_op_pauli_coeff
from ..entropy_randomized.exceptions import UnitaryOperatorNotFullCovering, MSG_FULL_COVER
from ...process.utils import qubit_mapper
from ...process.randomized_measure import generate_random_unitary
from ...tools import ParallelManager, set_pbar_description


def create_config(
    actual_qubits: int,
    measure: Optional[Union[list[int], tuple[int, int], int]],
    unitary_loc: Optional[Union[tuple[int, int], int]],
    which_circuit: Literal["1", "2"],
):
    """Create the configuration for the randomized measure.

    Args:
        actual_qubits (int): The number of qubits in the circuit.
        measure (Optional[Union[list[int], tuple[int, int], int]]):
            The selected qubits for the measurement.
            If it is None, then it will return the mapping of all qubits.
            If it is int, then it will return the mapping of the last n qubits.
            If it is tuple, then it will return the mapping of the qubits in the range.
            If it is list, then it will return the mapping of the selected qubits.
        unitary_loc (Optional[Union[tuple[int, int], int]]):
            The range of the unitary operator.
        which_circuit (Literal["1", "2"]): Which circuit this configuration belongs to.

    Returns:
        tuple[dict[int, int], list[int], dict[int, int], list[int]]:
            A tuple containing:

            - registers_mapping:
                The mapping of the index of selected qubits to the index of the classical register.
            - qubits_measured:
                The list of qubits that are measured.
            - unitary_located_mapping:
                The mapping of the index of unitary operator to the index of the classical register.
            - measured_but_not_unitary_located:
                The list of qubits that are measured but not located in the unitary operator.
    """

    registers_mapping = qubit_mapper(actual_qubits, measure)
    qubits_measured = list(registers_mapping)
    unitary_located_mapping = qubit_mapper(actual_qubits, unitary_loc)
    assert list(unitary_located_mapping.values()) == list(range(len(unitary_located_mapping))), (
        f"The unitary_located_mapping_{which_circuit} should be continuous."
    )
    measured_but_not_unitary_located = [
        qi for qi in qubits_measured if qi not in unitary_located_mapping
    ]

    return (
        registers_mapping,
        qubits_measured,
        unitary_located_mapping,
        measured_but_not_unitary_located,
    )


def overlapping_given_check(
    actual_qubits_1: int,
    actual_qubits_2: int,
    measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
    measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
    unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
    unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
):
    """Check whether the two circuits have overlapping qubits.

    Args:
        actual_qubits_1 (int): The number of qubits in the first circuit.
        actual_qubits_2 (int): The number of qubits in the second circuit.
        measure_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the first quantum circuit.
            If it is None, then it will return the mapping of all qubits.
            If it is int, then it will return the mapping of the last n qubits.
            If it is tuple, then it will return the mapping of the qubits in the range.
            If it is list, then it will return the mapping of the selected qubits.
            Defaults to None.
        measure_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the second quantum circuit.
            If it is None, then it will return the mapping of all qubits.
            If it is int, then it will return the mapping of the last n qubits.
            If it is tuple, then it will return the mapping of the qubits in the range.
            If it is list, then it will return the mapping of the selected qubits.
            Defaults to None.
        unitary_loc_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The range of the unitary operator for the first quantum circuit.
            Defaults to None.
        unitary_loc_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The range of the unitary operator for the second quantum circuit.
            Defaults to None.

    Raises:
        OverlapArgumentsUnfulfilled: If the number of qubits in the two circuits is not the same
            and the measure range or unitary location is not specified.
    """
    if actual_qubits_1 != actual_qubits_2:
        if any([measure_1 is None, measure_2 is None]):
            raise OverlapArgumentsUnfulfilled(MSG_OVERLAPPING_GIVEN.format("measure range"))
        if any([unitary_loc_1 is None, unitary_loc_2 is None]):
            raise OverlapArgumentsUnfulfilled(MSG_OVERLAPPING_GIVEN.format("unitary location"))


def overlapping_size_check(
    qubits_measured_1: list[int],
    qubits_measured_2: list[int],
    unitary_located_mapping_1: dict[int, int],
    unitary_located_mapping_2: dict[int, int],
):
    """Check whether the size of the qubits measured and unitary located mapping are the same.

    Args:
        qubits_measured_1 (list[int]): The qubits measured in the first circuit.
        qubits_measured_2 (list[int]): The qubits measured in the second circuit.
        unitary_located_mapping_1 (dict[int, int]):
            The unitary located mapping in the first circuit.
        unitary_located_mapping_2 (dict[int, int]):
            The unitary located mapping in the second circuit.

    Raises:
        OverlapComparisonSizeDifferent:
            If the size of the qubits measured or unitary located mapping
            in the two circuits are different.
    """

    if len(qubits_measured_1) != len(qubits_measured_2):
        raise OverlapComparisonSizeDifferent(
            NSG_OVERLAPPING_SIZE.format(
                "measuring range",
                len(qubits_measured_1),
                qubits_measured_1,
                len(qubits_measured_2),
                qubits_measured_2,
            )
        )
    if len(unitary_located_mapping_1) != len(unitary_located_mapping_2):
        raise OverlapComparisonSizeDifferent(
            NSG_OVERLAPPING_SIZE.format(
                "unitary location",
                len(unitary_located_mapping_1),
                unitary_located_mapping_1,
                len(unitary_located_mapping_2),
                unitary_located_mapping_2,
            )
        )


def unitary_full_cover_check(
    unitary_loc_not_cover_measure: bool,
    measured_but_not_unitary_located_1: list[int],
    measured_but_not_unitary_located_2: list[int],
    measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
    measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
    unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
    unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
):
    """Check whether the unitary operator covers the measurement.

    Args:
        unitary_loc_not_cover_measure (bool):
            If True, the unitary operator does not cover the measurement.
        measured_but_not_unitary_located_1 (list[int]):
            The qubits that are measured but not located in the first circuit.
        measured_but_not_unitary_located_2 (list[int]):
            The qubits that are measured but not located in the second circuit.
        measure_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the first quantum circuit.
            Defaults to None.
        measure_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the second quantum circuit.
            Defaults to None.
        unitary_loc_1 (Optional[Union[tuple[int, int], int]], optional):
            The range of the unitary operator for the first quantum circuit.
            Defaults to None.
        unitary_loc_2 (Optional[Union[tuple[int, int], int]], optional):
            The range of the unitary operator for the second quantum circuit.
            Defaults to None.

    Raises:
        RandomizedMeasureUnitaryOperatorNotFullCovering:
            If the unitary operator does not cover the measurement and
            `unitary_loc_not_cover_measure` is False.
    """

    if not unitary_loc_not_cover_measure:
        if measured_but_not_unitary_located_1:
            raise UnitaryOperatorNotFullCovering(
                MSG_FULL_COVER.format(
                    measured_but_not_unitary_located_1,
                    "first",
                    "unitary_loc_1",
                    unitary_loc_1,
                    "measure_1",
                    measure_1,
                ),
            )
        if measured_but_not_unitary_located_2:
            raise UnitaryOperatorNotFullCovering(
                MSG_FULL_COVER.format(
                    measured_but_not_unitary_located_2,
                    "second",
                    "unitary_loc_2",
                    unitary_loc_2,
                    "measure_2",
                    measure_2,
                ),
            )


def method_process(
    targets: list[tuple[WCKeyable, QuantumCircuit]],
    arguments: ELRArguments,
    pbar: Optional[tqdm.tqdm] = None,
    multiprocess: bool = False,
) -> tuple[list[QuantumCircuit], EntropyMeasureTalesTypes]:
    """The process method for building the circuits of the experiment.

    Args:
        targets (list[tuple[WCKeyable, QuantumCircuit]]):
            The circuits of the experiment.
        arguments (ELRArguments):
            The arguments of the experiment.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar for showing the progress of the experiment.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to False.

    Returns:
        A tuple containing a list of quantum circuits and a dictionary of additional information.
    """

    if len(targets) != 2:
        raise ValueError("The number of target circuits should be 2 for ELRExperiment.")
    target_key_1, target_circuit_1 = targets[0]
    target_key_1 = "" if isinstance(target_key_1, int) else str(target_key_1)
    target_key_2, target_circuit_2 = targets[1]
    target_key_2 = "" if isinstance(target_key_2, int) else str(target_key_2)

    set_pbar_description(pbar, f"Preparing {arguments.times} random unitary.")
    assert len(arguments.unitary_located_mapping_1) == len(arguments.unitary_located_mapping_2), (
        "The number of unitary_located_mapping_1 and "
        + "unitary_located_mapping_2 should be the same, "
        + f"but got {len(arguments.unitary_located_mapping_1)} "
        + f"and {len(arguments.unitary_located_mapping_2)}. "
        + "This should be ensured in the function 'params_control'."
    )
    unitary_dicts_source = generate_random_unitary(
        arguments.times,
        list(range(len(arguments.unitary_located_mapping_1))),
        arguments.random_unitary_seeds,
    )
    unitary_items = [
        (
            n_u_i,
            {
                qi: unitary_dicts_source[n_u_i][ui]
                for qi, ui in arguments.unitary_located_mapping_1.items()
            },
        )
        for n_u_i in range(arguments.times)
    ] + [
        (
            n_u_i + arguments.times,
            {
                qi: unitary_dicts_source[n_u_i][ui]
                for qi, ui in arguments.unitary_located_mapping_2.items()
            },
        )
        for n_u_i in range(arguments.times)
    ]
    unitary_items.sort(key=lambda x: x[0])
    unitary_dicts = dict(unitary_items)

    set_pbar_description(pbar, f"Building {arguments.times} circuits.")
    if multiprocess:
        pool = ParallelManager()
        circ_list = pool.starmap(
            make_samplied_circuit,
            [
                (
                    n_u_i,
                    target_circuit_1,
                    target_key_1,
                    arguments.exp_name,
                    arguments.registers_mapping_1,
                    unitary_dicts[n_u_i],
                )
                for n_u_i in range(arguments.times)
            ]
            + [
                (
                    n_u_i + arguments.times,
                    target_circuit_2,
                    target_key_2,
                    arguments.exp_name,
                    arguments.registers_mapping_2,
                    unitary_dicts[n_u_i + arguments.times],
                )
                for n_u_i in range(arguments.times)
            ],
        )
    else:
        circ_list = [
            make_samplied_circuit(
                n_u_i,
                target_circuit_1,
                target_key_1,
                arguments.exp_name,
                arguments.registers_mapping_1,
                unitary_dicts[n_u_i],
            )
            for n_u_i in range(arguments.times)
        ] + [
            make_samplied_circuit(
                n_u_i + arguments.times,
                target_circuit_2,
                target_key_2,
                arguments.exp_name,
                arguments.registers_mapping_2,
                unitary_dicts[n_u_i + arguments.times],
            )
            for n_u_i in range(arguments.times)
        ]
    other_results = [
        make_unitary_op_pauli_coeff(n_u_i, unitary_dicts[n_u_i]) for n_u_i in range(arguments.times)
    ]

    assert len(circ_list) == 2 * arguments.times, (
        "The number of circuits generated is not correct."
        + f" Get {len(circ_list)}, expect {2 * arguments.times}."
    )
    assert [x[0] for x in other_results] == list(range(arguments.times)), (
        "The indices of the results are not correct."
        + f" Get {[x[0] for x in other_results]}, expect {list(range(arguments.times))}."
    )

    return circ_list, {
        "unitary_operator": {i: u_op for i, u_op, _p_c in other_results},
        "bloch_vector": {i: p_c for i, _u_op, p_c in other_results},
    }


def process_duo_transpilation(
    circuits: list[QuantumCircuit],
    backend: Backend,
    transpile_args: TranspileArgs,
    passmanager_pair: Optional[tuple[str, PassManager]],
    second_backend: Backend,
    second_transpile_args: TranspileArgs,
    second_passmanager_pair: Optional[tuple[str, PassManager]],
    times: int,
    exp_id: str,
    multiprocess: bool = False,
    pbar: Optional[tqdm.tqdm] = None,
) -> list[QuantumCircuit]:
    """Process the transpilation of the circuits between 2 list of quantum circuits
    with respecting to the given backend and transpile arguments.

    Args:
        circuits (list[QuantumCircuit]):
            The circuits to be transpiled.
        backend (Backend):
            The backend to be used for transpilation.
        transpile_args (TranspileArgs):
            The transpile arguments.
        passmanager_pair (Optional[tuple[str, PassManager]]):
            The passmanager name and the passmanager to be used.
        second_backend (Backend):
            The backend to be used for transpilation of the second list of circuits.
        second_transpile_args (TranspileArgs):
            The transpile arguments of the second circuit.
        second_passmanager_pair (Optional[tuple[str, PassManager]]):
            The passmanager name and the passmanager to be used for the second list of circuits.
        times (int):
            The number of circuits for each quantum circuit.

        exp_id (str):
            The experiment ID, used for warning messages.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to False.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        list[QuantumCircuit]: The transpiled circuits.
    """
    if passmanager_pair is None:
        set_pbar_description(pbar, "Circuit transpiling...")
        transpile_args.pop("num_processes", None)
        transpiled_circs = transpile(
            circuits,
            backend=backend,
            num_processes=None if multiprocess else 1,
            **transpile_args,
        )
    else:
        passmanager_name, passmanager = passmanager_pair
        if not isinstance(passmanager, PassManager):
            raise TypeError(
                "The passmanager must be an instance of PassManager, "
                + f"not {type(passmanager)} in '{exp_id}'"
            )
        set_pbar_description(pbar, f"Circuit transpiling by passmanager '{passmanager_name}'...")
        transpiled_circs = passmanager.run(
            circuits=circuits[:times],
            num_processes=None if multiprocess else 1,  # type: ignore
        )
        if len(transpile_args) > 0:
            warnings.warn(
                f"Passmanager '{passmanager_name}' is given, "
                + f"the transpile_args will be ignored in '{exp_id}'",
                category=TranspileConfigurationIgnored,
            )

    return transpiled_circs
