"""ToolKits for Randomized Measure (:mod:`qurry.qurrium.utils.randomized`)

We also make a short import path for :func:`~qiskit.quantum_info.random_unitary`
as :func:`~qurry.qurrium.utils.randomized.random_unitary`,
due to Qiskit usually relocate its module.

"""

from typing import Union, Optional
import numpy as np

from qiskit.quantum_info import random_unitary, Operator

from .bloch_vector import qubit_operator_to_pauli_coeff


def generate_random_unitary(
    times: int,
    unitary_located: list[int],
    random_unitary_seeds: Optional[dict[int, dict[int, int]]],
) -> dict[int, dict[int, Operator]]:
    """Generate a dictionary of local random unitary operators.

    Args:
        times (int): The number of random unitary operators to generate.
        unitary_located (list[int]): The location of unitary operator.
        random_unitary_seeds (Optional[dict[int, dict[int, int]]]):
            The seeds for random unitary operator generation.

    Returns:
        dict[int, list[list[complex]]]:
            The dictionary of unitary operators in :class:`list[list[complex]]`.
    """

    if random_unitary_seeds is None:
        return {
            n_u_i: {n_u_qi: random_unitary(2) for n_u_qi in unitary_located}
            for n_u_i in range(times)
        }

    return {
        n_u_i: {
            n_u_qi: random_unitary(2, random_unitary_seeds[n_u_i][seed_i])
            for seed_i, n_u_qi in enumerate(unitary_located)
        }
        for n_u_i in range(times)
    }


def local_random_unitary_operators(
    unitary_loc: tuple[int, int],
    unitary_op_list: Union[list[np.ndarray], dict[int, Operator]],
) -> dict[int, list[list[complex]]]:
    """Transform a list of unitary operators in :class:`~qiskit.quantum_info.operator.Operator`
    into a list of unitary operators in :class:`list[list[complex]]`.

    Args:
        unitary_loc (tuple[int, int]): The location of unitary operator.
        unitary_op_list (Union[list[np.ndarray], dict[int, Operator]]):
            The list of unitary operators.

    Returns:
        The dictionary of unitary operators in :class:`list[list[complex]]`.
    """
    return {i: np.array(unitary_op_list[i]).tolist() for i in range(*unitary_loc)}


def local_random_unitary_pauli_coeff(
    unitary_loc: tuple[int, int],
    unitary_op_dict: dict[int, list[list[complex]]],
) -> dict[int, list[tuple[Union[float, np.float64], Union[float, np.float64]]]]:
    """Transform a list of unitary operators in :class:`~numpy.ndarray`
    into a list of pauli coefficients.

    Args:
        unitary_loc (tuple[int, int]): The location of unitary operator.
        unitary_op_dict (dict[int, list[list[complex]]]):
            The list of unitary operators or dictionary of unitary operators.

    Returns:
        The list of pauli coefficients.
    """
    return {i: qubit_operator_to_pauli_coeff(unitary_op_dict[i]) for i in range(*unitary_loc)}


def local_unitary_op_to_list(
    single_unitary_op_dict: dict[int, Operator],
) -> dict[int, list[list[complex]]]:
    """Transform a dictionary of local unitary operators
    in :class:`~qiskit.quantum_info.operator.Operator`
    with the qubit index as key to a dictionary of unitary operators
    in :class:`list[list[complex]]`.

    Args:
        single_unitary_op_dict (dict[int, Operator]): The dictionary of unitary operators.

    Returns:
        dict[int, list[list[complex]]]:
            The dictionary of unitary operators in :class:`list[list[complex]]`.
    """
    return {i: np.array(op).tolist() for i, op in single_unitary_op_dict.items()}


def local_unitary_op_to_pauli_coeff(
    single_unitary_op_list_dict: dict[int, list[list[complex]]],
) -> dict[int, list[tuple[Union[float, np.float64], Union[float, np.float64]]]]:
    """Transform a dictionary of local unitary operators in :class:`list[list[complex]]`
    with the qubit index as key to a dictionary of pauli coefficients.

    Args:
        single_unitary_dict (dict[int, Operator]): The dictionary of unitary operators.

    Returns:
        The dictionary of pauli coefficients.
    """
    return {
        i: qubit_operator_to_pauli_coeff(np.array(op))
        for i, op in single_unitary_op_list_dict.items()
    }
