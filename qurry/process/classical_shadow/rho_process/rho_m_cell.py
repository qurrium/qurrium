"""Post Processing - Classical Shadow - Rho Process - Rho M Cell
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_cell`)

"""

from typing import Literal, Union, Any, Sequence
import functools as ft
import numpy as np

from .unitary_set import (
    U_M_MATRIX,
    IDENTITY,
    OUTER_PRODUCT,
    cached_rho_m_k_i_matrix,
    cached_rho_m_k_i_matrix_2,
)

# pylint: disable=invalid-name
RhoMCellMethod = Union[Literal["numpy_proto", "numpy", "numpy_vectorized"], str]
"""Type for rho_m_cell method.
It can be either "numpy_proto", "numpy", or "numpy_vectorized".

- "numpy_proto": Use Numpy to calculate the rho_m.
- "numpy": Use Numpy to calculate the rho_m with precomputed values.
- "numpy_vectorized": Use Numpy to calculate the rho_m with a flattening workflow.

Currently, "numpy" is the best option for performance.
"""
# pylint: enable=invalid-name


def rho_m_cell_prototype(
    single_counts: dict[str, int],
    single_random_basis: dict[int, Union[Literal[0, 1, 2], int]],
    selected_clregs_sorted: list[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    r""":math:`\rho_{m}` calculation from single counts.

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
        single_random_basis (dict[int, Union[Literal[0, 1, 2], int]]):
            The shadow direction of the unitary operators.
        selected_clregs_sorted (list[int]):
            The **sorted** list of **the index of the selected_classical_registers**.

    Returns:
        The rho_m.
    """

    n_qubits = len(selected_clregs_sorted)
    matrix_dim = 2**n_qubits

    if not single_counts:
        return np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)

    # core calculation
    rho_m_k_data: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = []

    for bitstring in single_counts.keys():
        tmp_dict = {
            q_di: (
                3
                * U_M_MATRIX[single_random_basis[q_di]].conj().T
                @ OUTER_PRODUCT[s_q]
                @ U_M_MATRIX[single_random_basis[q_di]]
            )
            - IDENTITY
            for q_di, s_q in zip(selected_clregs_sorted, bitstring)
        }
        tmp: Any = tmp_dict[selected_clregs_sorted[0]]
        for q_di in selected_clregs_sorted[1:]:
            tmp = np.kron(tmp, tmp_dict[q_di])

        rho_m_k_data.append(tmp)

    counts_nums = list(single_counts.values())

    return np.average(rho_m_k_data, axis=0, weights=counts_nums)


def rho_m_cell_precomputed(
    single_counts: dict[str, int],
    single_random_basis: dict[int, Union[Literal[0, 1, 2], int]],
    selected_clregs_sorted: list[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
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
        single_random_basis (dict[int, Union[Literal[0, 1, 2], int]]):
            The shadow direction of the unitary operators.
        selected_clregs_sorted (list[int]):
            The **sorted** list of **the index of the selected_classical_registers**.

    Returns:
        The rho_m.
    """

    n_qubits = len(selected_clregs_sorted)
    matrix_dim = 2**n_qubits

    if not single_counts:
        return np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)

    bitstrings = list(single_counts.keys())
    counts_nums = list(single_counts.values())

    single_matrices = np.empty((len(bitstrings), n_qubits), dtype=object)
    for i, bitstring in enumerate(bitstrings):
        for j, (q_di, s_q) in enumerate(zip(selected_clregs_sorted, bitstring)):
            single_matrices[i, j] = cached_rho_m_k_i_matrix(single_random_basis[q_di], s_q)

    all_rho_mk = np.empty((len(bitstrings), matrix_dim, matrix_dim), dtype=np.complex128)
    for i in range(len(bitstrings)):
        all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])

    return np.average(all_rho_mk, axis=0, weights=counts_nums)


def rho_m_cell_vectorized(
    seq_rho_mki_kinds: Sequence[Sequence[int]], count_num: Sequence[int]
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
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

    Returns:
        The rho_m.
    """

    bits_array_np = np.asarray(seq_rho_mki_kinds)
    n_samples, n_qubits = bits_array_np.shape
    matrix_dim = 2**n_qubits

    single_matrices = np.vectorize(cached_rho_m_k_i_matrix_2, otypes=[object])(bits_array_np)

    all_rho_mk = np.empty((n_samples, matrix_dim, matrix_dim), dtype=np.complex128)
    for i in range(n_samples):
        all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])

    return np.average(all_rho_mk, axis=0, weights=count_num)
