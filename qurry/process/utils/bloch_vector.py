"""Bloch Vector Utilities (:mod:`qurry.process.utils.bloch_vector`)"""

from typing import Union, Sequence
import numpy as np
import numpy.typing as npt

PauliXmatrix: npt.NDArray[np.complex128] = np.array([[0, 1], [1, 0]])
"""Pauli-X matrix"""
PauliYmatrix: npt.NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]])
"""Pauli-Y matrix"""
PauliZmatrix: npt.NDArray[np.complex128] = np.array([[1, 0], [0, -1]])
"""Pauli-Z matrix"""


def density_matrix_to_bloch_vector(
    rho: Union[npt.NDArray[np.complex128], Sequence[Sequence[complex]]],
) -> tuple[float, float, float]:
    """Convert a density matrix to a Bloch vector.

    Args:
        rho (Union[npt.NDArray[np.complex128], Sequence[Sequence[complex]]]):
            The density matrix.
            It can be a :class:`~numpy.ndarray` or :class:`list[list[complex]]`.
            The matrix should be a 2x2 matrix.

    Returns:
        The bloch vector.
    """

    return (
        float(np.trace(np.dot(rho, PauliXmatrix)).real),
        float(np.trace(np.dot(rho, PauliYmatrix)).real),
        float(np.trace(np.dot(rho, PauliZmatrix)).real),
    )
