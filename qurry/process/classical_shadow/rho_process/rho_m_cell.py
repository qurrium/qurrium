"""Post Processing - Classical Shadow - Rho Process - Rho M Cell
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_cell`)

"""

from typing import Literal
from collections.abc import Sequence
import functools as ft
import numpy as np
import numpy.typing as npt

from .unitary_set import ShadowRandomBasis

# pylint: disable=invalid-name
RhoMCellMethod = Literal["numpy", "numpy_vectorized"] | str
"""Type for rho_m_cell method.
It can be either "numpy" or "numpy_vectorized".

- "numpy": Use Numpy to calculate the rho_m with precomputed values.
- "numpy_vectorized": Use Numpy to calculate the rho_m with a flattening workflow.

Currently, "numpy" is the best option for performance.
"""
# pylint: enable=invalid-name


def rho_m_cell_precomputed(
    single_counts: dict[str, int],
    single_random_basis: list[int],
    selected_clregs_sorted: list[int],
    random_basis_obj: ShadowRandomBasis,
) -> npt.NDArray[np.complex128]:
    r""":math:`\rho_{m}` calculation from single counts with pre-computed.

    The matrix :math:`\rho_{mk}^{i}` is calculated by the following equation,

    .. math::
        \rho_{mk}^{i} = \frac{3} U_{mi}^{\dagger} |b_k \rangle\langle b_k | U_{mi} - \mathbb{1}

    The matrix :math:`\rho_{mk}` is calculated by the following equation,

    .. math::
        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    .. math::
        \rho_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} \rho_{mk}

    where :math:`N_M` is the number of shots.

    Args:
        single_counts (dict[str, int]):
            Counts measured by the single quantum circuit.
        single_random_basis (list[int]):
            The shadow direction of the unitary operators.
        selected_clregs_sorted (list[int]):
            The **sorted** list of **the index of the selected_classical_registers**.
        random_basis_obj (ShadowRandomBasis):
            The :class:`~.ShadowRandomBasis` object.
            Provides the precomputed :math:`\rho_{mk}^{i}` matrix.

    Returns:
        The rho_m.
    """

    n_qubits = len(selected_clregs_sorted)
    matrix_dim = 2**n_qubits

    if not single_counts:
        return np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)

    bitstrings = list(single_counts.keys())
    counts_nums = list(single_counts.values())

    precomputed_rho_m_k_i = random_basis_obj.basis_precomputed_rho_m_k_i

    single_matrices = np.empty((len(bitstrings), n_qubits), dtype=object)
    for i, bitstring in enumerate(bitstrings):
        for j, (c_i, s_b) in enumerate(zip(selected_clregs_sorted, bitstring)):
            # The order of classical registers is [8, 7, 6, 5, 4, 3, 2, 1, 0]
            # which respects to the bitstring "000000000"
            single_matrices[i, j] = precomputed_rho_m_k_i[(single_random_basis[c_i], s_b)]

    all_rho_mk = np.empty((len(bitstrings), matrix_dim, matrix_dim), dtype=np.complex128)
    for i in range(len(bitstrings)):
        all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])

    return np.average(all_rho_mk, axis=0, weights=counts_nums)


def rho_m_cell_vectorized(
    seq_rho_mki_kinds: Sequence[Sequence[int]],
    count_num: Sequence[int],
    random_basis_obj: ShadowRandomBasis,
) -> npt.NDArray[np.complex128]:
    r""":math:`\rho_{m}` calculation from single counts with a vectorized workflow.

    The matrix :math:`\rho_{mk}^{i}` is calculated by the following equation,

    .. math::
        \rho_{mk}^{i} = \frac{3} U_{mi}^{\dagger} |b_k \rangle\langle b_k | U_{mi} - \mathbb{1}

    The matrix :math:`\rho_{mk}` is calculated by the following equation,

    .. math::
        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    .. math::
        \rho_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} \rho_{mk}

    where :math:`N_M` is the number of shots.

    Args:
        seq_rho_mki_kinds (Sequence[Sequence[int]]):
            The sequence of sequence of the kinds of rho_m_k_i.
        count_num (Sequence[int]): The counts for each bitstring.
        random_basis_obj (ShadowRandomBasis):
            The :class:`~.ShadowRandomBasis` object.
            Provides the precomputed :math:`\rho_{mk}^{i}` matrix.

    Returns:
        The rho_m.
    """

    bits_array_np = np.asarray(seq_rho_mki_kinds)
    n_samples, n_qubits = bits_array_np.shape
    matrix_dim = 2**n_qubits

    basis_precomputed_rho_m_k_i_2 = random_basis_obj.basis_precomputed_rho_m_k_i_2

    single_matrices = np.vectorize(
        lambda direction_and_b_k: basis_precomputed_rho_m_k_i_2[direction_and_b_k], otypes=[object]
    )(bits_array_np)

    all_rho_mk = np.empty((n_samples, matrix_dim, matrix_dim), dtype=np.complex128)
    for i in range(n_samples):
        all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])

    return np.average(all_rho_mk, axis=0, weights=count_num)
