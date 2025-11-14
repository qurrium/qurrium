"""Bloch Vector Utilities (:mod:`qurry.process.utils.bloch_vector`)"""

from typing import Union
import numpy as np

PauliXmatrix: np.ndarray[tuple[int, ...], np.dtype[np.complex128]] = np.array([[0, 1], [1, 0]])
"""Pauli-X matrix"""
PauliYmatrix: np.ndarray[tuple[int, ...], np.dtype[np.complex128]] = np.array([[0, -1j], [1j, 0]])
"""Pauli-Y matrix"""
PauliZmatrix: np.ndarray[tuple[int, ...], np.dtype[np.complex128]] = np.array([[1, 0], [0, -1]])
"""Pauli-Z matrix"""


def density_matrix_to_bloch(
    rho: Union[np.ndarray[tuple[int, ...], np.dtype[np.complex128]], list[list[complex]]],
) -> list[float]:
    """Convert a density matrix to a Bloch vector.

    Args:
        rho (Union[
            np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
            list[list[complex]]
        ]):
            The density matrix.
            It can be a :class:`~numpy.ndarray` or :class:`list[list[complex]]`.
            The matrix should be a 2x2 matrix.

    Returns:
        list[float]: The bloch vector.
    """

    ax = np.trace(np.dot(rho, PauliXmatrix)).real
    ay = np.trace(np.dot(rho, PauliYmatrix)).real
    az = np.trace(np.dot(rho, PauliZmatrix)).real
    return [ax, ay, az]


def qubit_operator_to_pauli_coeff(
    rho: Union[np.ndarray[tuple[int, ...], np.dtype[np.complex128]], list[list[complex]]],
) -> list[tuple[Union[float, np.float64], Union[float, np.float64]]]:
    """Convert a random unitary operator matrix to a Bloch vector.

    Args:
        rho (Union[
            np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
            list[list[complex]]
        ]):
            The random unitary operator matrix.
            It can be a :class:`~numpy.ndarray` or :class:`list[list[complex]]`.
            The matrix should be a 2x2 matrix.

    Returns:
        list[tuple[float]]: The bloch vector divided as tuple of real number and image number.
    """

    ax = np.trace(np.dot(rho, PauliXmatrix)) / 2
    ay = np.trace(np.dot(rho, PauliYmatrix)) / 2
    az = np.trace(np.dot(rho, PauliZmatrix)) / 2
    return [(np.float64(a.real), np.float64(a.imag)) for a in [ax, ay, az]]
