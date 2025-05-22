"""Post Processing - Classical Shadow - Rho M K Cell
(:mod:`qurry.process.classical_shadow.rho_mk_cell`)

"""

from typing import Literal, Union, Any
import numpy as np

from .unitary_set import U_M_MATRIX, IDENTITY, OUTER_PRODUCT
from .matrix_calcution import rho_mki_kronecker_product_numpy
from ..utils import single_counts_recount_pyrust

# pylint: disable=invalid-name
RhoMKCellMethod = Union[Literal["numpy", "numpy_precomputed"], str]
"""Type for rho_mk_cell method.
It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
- "numpy": Use Numpy to calculate the rho_m.
- "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
Currently, "numpy_precomputed" is the best option for performance.
"""
# pylint: enable=invalid-name


def rho_mk_cell_py(
    idx: int,
    single_counts: dict[str, int],
    nu_shadow_direction: dict[int, Union[Literal[0, 1, 2], int]],
    selected_classical_registers: list[int],
) -> tuple[
    int,
    list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]],
    list[int],
]:
    r""":math:`\rho_{mk}` calculation for single cell.

    The matrix :math:`\rho_{mk}^{i} is calculated by the following equation,
    .. math::

        \rho_{mk}^{i} = \frac{3} U_{mi}^{\dagger} |b_k \rangle\langle b_k | U_{mi} - \mathbb{1}

    The matrix :math:`\rho_{mk}` is calculated by the following equation,
    .. math::

        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    Args:
        idx (int):
            Index of the cell (counts).
        single_counts (dict[str, int]):
            Counts measured by the single quantum circuit.
        nu_shadow_direction (dict[int, Union[Literal[0, 1, 2], int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[
            int,
            list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]
            list[int],
        ]:
            Index, the list of the data of rho_mk, the sorted list of the selected qubits
    """

    # subsystem making
    num_classical_register = len(list(single_counts.keys())[0])
    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    single_counts_under_degree = single_counts_recount_pyrust(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )

    # core calculation
    rho_m_k_data: list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]] = []

    for bitstring, num_bitstring in single_counts_under_degree.items():
        tmp_dict = {
            q_di: (
                3
                * U_M_MATRIX[nu_shadow_direction[q_di]].conj().T
                @ OUTER_PRODUCT[s_q]
                @ U_M_MATRIX[nu_shadow_direction[q_di]]
            )
            - IDENTITY
            for q_di, s_q in zip(selected_classical_registers_sorted, bitstring)
        }
        tmp: Any = tmp_dict[selected_classical_registers_sorted[0]]
        for q_di in selected_classical_registers_sorted[1:]:
            tmp = np.kron(tmp, tmp_dict[q_di])

        rho_m_k_data.append((bitstring, num_bitstring, tmp))

    return idx, rho_m_k_data, selected_classical_registers_sorted


def rho_mk_cell_py_precomputed(
    idx: int,
    single_counts: dict[str, int],
    nu_shadow_direction: dict[int, Union[Literal[0, 1, 2], int]],
    selected_classical_registers: list[int],
) -> tuple[
    int,
    list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]],
    list[int],
]:
    r""":math:`\rho_{mk}` calculation for single cell.

    The matrix :math:`\rho_{mk}^{i} is calculated by the following equation,
    .. math::

        \rho_{mk}^{i} = \frac{3} U_{mi}^{\dagger} |b_k \rangle\langle b_k | U_{mi} - \mathbb{1}

    The matrix :math:`\rho_{mk}` is calculated by the following equation,
    .. math::

        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    Args:
        idx (int):
            Index of the cell (counts).
        single_counts (dict[str, int]):
            Counts measured by the single quantum circuit.
        nu_shadow_direction (dict[int, Union[Literal[0, 1, 2], int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[
            int,
            list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]
            list[int],
        ]:
            Index, the list of the data of rho_mk, the sorted list of the selected qubits
    """

    # subsystem making
    num_classical_register = len(list(single_counts.keys())[0])
    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    single_counts_under_degree = single_counts_recount_pyrust(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )

    # core calculation
    rho_m_k_data: list[tuple[str, int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]] = []
    for bitstring, num_bitstring in single_counts_under_degree.items():
        tmp = rho_mki_kronecker_product_numpy(
            [
                (nu_shadow_direction[q_di], s_q)
                for q_di, s_q in zip(selected_classical_registers_sorted, bitstring)
            ]
        )
        rho_m_k_data.append((bitstring, num_bitstring, tmp))

    return idx, rho_m_k_data, selected_classical_registers_sorted
