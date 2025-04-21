"""ToolKits for Randomized Measure (:mod:`qurry.qurrium.utils.randomized`)"""

from typing import Union, Literal
import numpy as np

# pylint: disable=unused-import
from qiskit.quantum_info import random_unitary, Operator

# pylint: enable=unused-import

RXmatrix: np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]] = np.array(
    [[0, 1], [1, 0]]
)
"""Pauli-X matrix"""
RYmatrix: np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]] = np.array(
    [[0, -1j], [1j, 0]]
)
"""Pauli-Y matrix"""
RZmatrix: np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]] = np.array(
    [[1, 0], [0, -1]]
)
"""Pauli-Z matrix"""


def density_matrix_to_bloch(
    rho: Union[
        np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]], list[list[complex]]
    ],
) -> list[float]:
    """Convert a density matrix to a Bloch vector.

    Args:
        rho (Union[
            np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]],
            list[list[complex]]
        ]):
            The density matrix.
            It can be a :cls:`numpy.ndarray` or :cls:`list[list[complex]]`.
            The matrix should be a 2x2 matrix.

    Returns:
        list[float]: The bloch vector.
    """

    ax = np.trace(np.dot(rho, RXmatrix)).real
    ay = np.trace(np.dot(rho, RYmatrix)).real
    az = np.trace(np.dot(rho, RZmatrix)).real
    return [ax, ay, az]


def qubit_operator_to_pauli_coeff(
    rho: Union[
        np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]], list[list[complex]]
    ],
) -> list[tuple[Union[float, np.float64], Union[float, np.float64]]]:
    """Convert a random unitary operator matrix to a Bloch vector.

    Args:
        rho (Union[
            np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]],
            list[list[complex]]
        ]):
            The random unitary operator matrix.
            It can be a :cls:`numpy.ndarray` or :cls:`list[list[complex]]`.
            The matrix should be a 2x2 matrix.

    Returns:
        list[tuple[float]]: The bloch vector divided as tuple of real number and image number.
    """

    ax = np.trace(np.dot(rho, RXmatrix)) / 2
    ay = np.trace(np.dot(rho, RYmatrix)) / 2
    az = np.trace(np.dot(rho, RZmatrix)) / 2
    return [(np.float64(a.real), np.float64(a.imag)) for a in [ax, ay, az]]


def local_random_unitary_operators(
    unitary_loc: tuple[int, int],
    unitary_op_list: Union[list[np.ndarray], dict[int, Operator]],
) -> dict[int, list[list[complex]]]:
    """Transform a list of unitary operators in :cls:`qiskit.quantum_info.operator.Operator`
    a list of unitary operators in :cls:`numpy.ndarray`.

    Args:
        unitary_loc (tuple[int, int]): The location of unitary operator.
        unitary_op_list (Union[list[np.ndarray], dict[int, Operator]]):
            The list of unitary operators.

    Returns:
        dict[int, list[list[complex]]]:
            The dictionary of unitary operators in `list[list[complex]]`.
    """
    return {i: np.array(unitary_op_list[i]).tolist() for i in range(*unitary_loc)}


def local_random_unitary_pauli_coeff(
    unitary_loc: tuple[int, int],
    unitary_op_dict: dict[int, list[list[complex]]],
) -> dict[int, list[tuple[Union[float, np.float64], Union[float, np.float64]]]]:
    """Transform a list of unitary operators in :cls:`numpy.ndarray`
    a list of pauli coefficients.

    Args:
        unitary_loc (tuple[int, int]): The location of unitary operator.
        unitary_op_dict (dict[int, list[list[complex]]]):
            The list of unitary operators or dictionary of unitary operators.

    Returns:
        dict[int, list[tuple[float, float]]]: The list of pauli coefficients.
    """
    return {i: qubit_operator_to_pauli_coeff(unitary_op_dict[i]) for i in range(*unitary_loc)}


def local_unitary_op_to_list(
    single_unitary_op_dict: dict[int, Operator],
) -> dict[int, list[list[complex]]]:
    """Transform a dictionary of local unitary operators
    in :cls:`qiskit.quantum_info.operator.Operator`
    with the qubit index as key to a dictionary of unitary operators in :cls:`list[list[complex]]`.

    Args:
        single_unitary_op_dict (dict[int, Operator]): The dictionary of unitary operators.

    Returns:
        dict[int, list[list[complex]]]:
            The dictionary of unitary operators in :cls:`list[list[complex]]`.
    """
    return {i: np.array(op).tolist() for i, op in single_unitary_op_dict.items()}


def local_unitary_op_to_pauli_coeff(
    single_unitary_op_list_dict: dict[int, list[list[complex]]],
) -> dict[int, list[tuple[Union[float, np.float64], Union[float, np.float64]]]]:
    """Transform a dictionary of local unitary operators in :cls:`list[list[complex]]`
    with the qubit index as key to a dictionary of pauli coefficients.

    Args:
        single_unitary_dict (dict[int, Operator]): The dictionary of unitary operators.

    Returns:
        dict[int, list[tuple[float, float]]]: The dictionary of pauli coefficients.
    """
    return {
        i: qubit_operator_to_pauli_coeff(np.array(op))
        for i, op in single_unitary_op_list_dict.items()
    }
