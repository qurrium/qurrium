"""Post Processing - Classical Shadow - Rho Process - Rho M Cell
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_cell`)

"""

from typing import Literal
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt

from .unitary_set import ShadowRandomBasis

RhoMCellMethod = Literal["numpy", "numpy_vectorized"] | str
"""Type for rho_m_cell method.
It can be either "numpy" or "numpy_vectorized".

- "numpy": Use Numpy to calculate the rho_m with precomputed values.
- "numpy_vectorized": Use Numpy to calculate the rho_m with a flattening workflow.

Currently, "numpy" is the best option for performance.
"""


def kron_rho_mk_batch(
    single_matrices: np.ndarray[tuple[int, int, int, int], np.dtype[np.complex128]],
    counts: Sequence[int] | npt.NDArray[np.int_],
) -> npt.NDArray[np.complex128]:
    r"""Batched Kronecker product with weighted average — best pure-Python implementation.

    Replaces the per-sample Python loop::

        all_rho_mk = np.empty((n_samples, matrix_dim, matrix_dim), dtype=np.complex128)
        # or
        # all_rho_mk = np.empty((len(bitstrings), matrix_dim, matrix_dim), dtype=np.complex128)
        for i in range(n_samples):
            all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])
        return np.average(all_rho_mk, axis=0, weights=counts)

    The outer loop runs only *n_qubits* times (typically ≤ 20).
    All n_samples are handled simultaneously in C via ``np.einsum``.

    The identity used is:

    .. math::

        (A \otimes B)_{ik,jl} = A_{ij} B_{kl}

    implemented as ``einsum('sij,skl->sikjl', result, mats[:, q])`` plus a reshape
    that merges the row-pair ``(i, k)`` and col-pair ``(j, l)``.

    .. note::
        This function is made by Claude Sonnet 4.6 on GitHub Copilot with minor modifications.

    Args:
        single_matrices: Shape ``(n_samples, n_qubits, 2, 2)``, dtype ``complex128``.
        counts: Per-sample weights (measurement counts), length ``n_samples``.

    Returns:
        Weighted-average :math:`\rho_m`, shape ``(2**n_qubits, 2**n_qubits)``,
        dtype ``complex128``.
    """
    if single_matrices.ndim != 4 or single_matrices.shape[2:] != (2, 2):
        raise ValueError(
            f"single_matrices must have shape (n_samples, n_qubits, 2, 2), "
            f"but got {single_matrices.shape}"
        )

    # Batched sequential Kronecker product.
    # At step j, `result` has shape (n_samples, 2^j, 2^j).
    # The einsum computes the outer product for all samples at once;
    # reshape then merges the index pairs to produce (n_samples, 2^(j+1), 2^(j+1)).
    n_samples, n_qubits, _, _ = single_matrices.shape
    result: npt.NDArray[np.complex128] = single_matrices[:, 0]  # (n_samples, 2, 2)
    for j in range(1, n_qubits):
        dim = result.shape[1]
        result = np.einsum("sij,skl->sikjl", result, single_matrices[:, j]).reshape(
            n_samples, dim * 2, dim * 2
        )

    return np.average(result, axis=0, weights=counts)


def rho_m_cell_precomputed(
    single_counts: dict[str, int],
    single_random_basis: list[int],
    selected_clregs_sorted: list[int],
    random_basis_obj: ShadowRandomBasis,
    use_projecter: bool = False,
) -> npt.NDArray[np.complex128]:
    r""":math:`\rho_{m}` calculation from single counts with pre-computed.

    The matrix :math:`\rho_{mk}^{i}` is calculated by the following equation,

    .. math::
        P_{mk}^{i} = U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} \\
        \rho_{mk}^{i} = 3 P_{mk}^{i} - \mathbb{I}

    with projecter :math:`P_{mk}^{i}`

    The matrix :math:`\rho_{mk}` is calculated by the following equation,

    .. math::
        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    .. math::
        \rho_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} \rho_{mk}

    where :math:`N_M` is the number of shots.
    
    When :attr:`use_projecter` is set to True,
    it will give the results :math:`P_m` using the projecter :math:`P_{mk}^{i}`
    instead of the precomputed :math:`\rho_{mk}^{i}` matrix.
    which is calculated by the following equation,

    .. math::
        P_{mk}^{i} = U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} \\
        P_{mk} = \bigotimes_{i=1}^{N_q} P_{mk}^{i} \\
        P_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} P_{mk}

    Args:
        single_counts (dict[str, int]):
            Counts measured by the single quantum circuit.
        single_random_basis (list[int]):
            The shadow direction of the unitary operators.
        selected_clregs_sorted (list[int]):
            The **sorted** list of **the index of the selected_classical_registers**.
        random_basis_obj (ShadowRandomBasis):
            The :class:`~.ShadowRandomBasis` object.
            Provides the precomputed :math:`\rho_{mk}^{i}` and projecter :math:`P_{mk}^{i}`.
        use_projecter (bool):
            Whether to use the projecter :math:`P_{mk}^{i}`
            instead of the precomputed :math:`\rho_{mk}^{i}`.
            Default is False, which means using the precomputed :math:`\rho_{mk}^{i}`.

    Returns:
        The :math:`\rho_{m}` or :math:`P_m` depending on the value of :attr:`use_projecter`.
    """

    n_qubits = len(selected_clregs_sorted)
    matrix_dim = 2**n_qubits

    if not single_counts:
        return np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)

    bitstrings = list(single_counts.keys())
    counts_nums = list(single_counts.values())

    unit_m_k_i = (
        random_basis_obj.basis_projecters
        if use_projecter
        else random_basis_obj.basis_precomputed_rho_m_k_i
    )

    single_matrices: np.ndarray[tuple[int, int, int, int], np.dtype[np.complex128]] = np.empty(
        (len(bitstrings), n_qubits, 2, 2), dtype=np.complex128
    )
    for i, bitstring in enumerate(bitstrings):
        for j, (c_i, s_b) in enumerate(zip(selected_clregs_sorted, bitstring)):
            # The order of classical registers is [8, 7, 6, 5, 4, 3, 2, 1, 0]
            # which respects to the bitstring "000000000"
            single_matrices[i, j] = unit_m_k_i[(single_random_basis[c_i], s_b)]

    assert single_matrices.shape == (len(bitstrings), n_qubits, 2, 2), (
        f"single_matrices.shape: {single_matrices.shape}, "
        + f"expected: {(len(bitstrings), n_qubits, 2, 2)}"
    )

    return kron_rho_mk_batch(single_matrices, counts_nums)


def rho_m_cell_vectorized(
    seq_rho_mki_kinds: Sequence[Sequence[int]],
    count_num: Sequence[int],
    random_basis_obj: ShadowRandomBasis,
    use_projecter: bool = False,
) -> npt.NDArray[np.complex128]:
    r""":math:`\rho_{m}` calculation from single counts with a vectorized workflow.

    The matrix :math:`\rho_{mk}^{i}` is calculated by the following equation,

    .. math::
        P_{mk}^{i} = U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} \\
        \rho_{mk}^{i} = 3 P_{mk}^{i} - \mathbb{I}

    with projecter :math:`P_{mk}^{i}`

    The matrix :math:`\rho_{mk}` is calculated by the following equation,

    .. math::
        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    .. math::
        \rho_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} \rho_{mk}

    where :math:`N_M` is the number of shots.

    When :attr:`use_projecter` is set to True,
    it will give the results :math:`P_m` using the projecter :math:`P_{mk}^{i}`
    instead of the precomputed :math:`\rho_{mk}^{i}` matrix.
    which is calculated by the following equation,
    
    .. math::
        P_{mk}^{i} = U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} \\
        P_{mk} = \bigotimes_{i=1}^{N_q} P_{mk}^{i} \\
        P_{m} = \frac{1}{N_M} \sum_{k = 1}^{N_M} P_{mk}

    Args:
        seq_rho_mki_kinds (Sequence[Sequence[int]]):
            The sequence of sequence of the kinds of rho_m_k_i.
        count_num (Sequence[int]): The counts for each bitstring.
        random_basis_obj (ShadowRandomBasis):
            The :class:`~.ShadowRandomBasis` object.
            Provides the precomputed :math:`\rho_{mk}^{i}` and projecter :math:`P_{mk}^{i}`.
        use_projecter (bool):
            Whether to use the projecter :math:`P_{mk}^{i}`
            instead of the precomputed :math:`\rho_{mk}^{i}`.
            Default is False, which means using the precomputed :math:`\rho_{mk}^{i}`.

    Returns:
        The :math:`\rho_{m}` or :math:`P_m` depending on the value of :attr:`use_projecter`.
    """

    bits_array_np = np.asarray(seq_rho_mki_kinds, dtype=np.int8)

    unit_m_k_i_2 = (
        random_basis_obj.basis_projecters_2
        if use_projecter
        else random_basis_obj.basis_precomputed_rho_m_k_i_2
    )

    single_matrices: np.ndarray[tuple[int, int, int, int], np.dtype[np.complex128]] = np.vectorize(
        lambda direction_and_b_k: unit_m_k_i_2[direction_and_b_k],
        signature="()->(2,2)",
    )(bits_array_np)

    assert single_matrices.shape == (*bits_array_np.shape, 2, 2), (
        f"single_matrices.shape: {single_matrices.shape}, "
        + f"expected: {(*bits_array_np.shape, 2, 2)}"
    )

    return kron_rho_mk_batch(single_matrices, count_num)
