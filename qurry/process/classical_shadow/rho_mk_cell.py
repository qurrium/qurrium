"""Post Processing - Classical Shadow - Rho M K Cell
(:mod:`qurry.process.classical_shadow.rho_mk_cell`)

"""

import warnings
from typing import Literal, Union, Any, Iterable
import numpy as np

from .unitary_set import U_M_MATRIX, OUTER_PRODUCT, IDENTITY
from ..utils import counts_under_degree_pyrust


def rho_mk_cell_py(
    idx: int,
    single_counts: dict[str, int],
    nu_shadow_direction: dict[int, Union[Literal[0, 1, 2], int]],
    selected_classical_registers: list[int],
) -> tuple[
    int,
    dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    dict[str, int],
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
            dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            dict[str, int],
            list[int],
        ]:
            Index, rho_mk, the sorted list of the selected qubits
    """

    # subsystem making
    num_classical_register = len(list(single_counts.keys())[0])
    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    single_counts_under_degree = counts_under_degree_pyrust(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )

    # core calculation
    rho_m_k_i_dict: dict[
        str, dict[int, np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]]]
    ] = {}
    rho_m_k_dict: dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = {}
    rho_m_k_counts_num: dict[str, int] = {}

    for bitstring, num_bitstring in single_counts_under_degree.items():
        rho_m_k_i_dict[bitstring] = {}
        rho_m_k_counts_num[bitstring] = num_bitstring
        for q_di, s_q in zip(selected_classical_registers_sorted, bitstring):
            rho_m_k_i_dict[bitstring][q_di] = (
                3
                * U_M_MATRIX[nu_shadow_direction[q_di]].conj().T
                @ OUTER_PRODUCT[s_q]
                @ U_M_MATRIX[nu_shadow_direction[q_di]]
            ) - IDENTITY

        tmp: Any = rho_m_k_i_dict[bitstring][selected_classical_registers_sorted[0]]
        for q_di in selected_classical_registers_sorted[1:]:
            tmp = np.kron(tmp, rho_m_k_i_dict[bitstring][q_di])
        rho_m_k_dict[bitstring] = tmp

    return idx, rho_m_k_dict, rho_m_k_counts_num, selected_classical_registers_sorted


def rho_mk_cell_py_wrapper(
    arguments: tuple[
        int,
        dict[str, int],
        dict[int, Union[Literal[0, 1, 2], int]],
        list[int],
    ],
) -> tuple[
    int,
    dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    dict[str, int],
    list[int],
]:
    """Wrapper for rho_mk_cell_py.

    Args:
        arguments (tuple):
            The arguments for rho_mk_cell_py.

    Returns:
        tuple[
            int,
            dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            dict[str, int],
            list[int],
        ]:
            The result of rho_mk_cell_py.
    """
    return rho_mk_cell_py(*arguments)


def handle_rho_mk_py_iterable(
    rho_mk_py_result_iterable: Iterable[
        tuple[
            int,
            dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            dict[str, int],
            list[int],
        ]
    ],
    shots: int,
    selected_classical_registers_sorted: list[int],
) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
    """Handle rho_mk_py_result_iterable.

    Args:
        rho_mk_py_result_iterable (Iterable):
            The iterable of rho_mk_py_result.
        shots (int):
            The number of shots.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
            The dictionary of rho_mk.
    """
    rho_m_dict = {}
    selected_qubits_checked: dict[int, bool] = {}

    for (
        idx,
        rho_mk_dict,
        rho_mk_counts_num,
        selected_classical_registers_sorted_result,
    ) in rho_mk_py_result_iterable:

        tmp_arr = np.array(
            [rho_mk * rho_mk_counts_num[bitstring] for bitstring, rho_mk in rho_mk_dict.items()]
        )
        tmp = tmp_arr.sum(axis=0, dtype=np.complex128)
        tmp /= shots
        expected_shape = (
            2 ** len(selected_classical_registers_sorted),
            2 ** len(selected_classical_registers_sorted),
        )
        assert (
            tmp.shape == expected_shape
        ), f"Invalid rho_m shape {tmp.shape}, expected {expected_shape} for {idx} cell."
        rho_m_dict[idx] = tmp
        selected_qubits_checked[idx] = (
            selected_classical_registers_sorted_result != selected_classical_registers_sorted
        )

    if any(selected_qubits_checked.values()):
        problematic_cells = [idx for idx, checked in selected_qubits_checked.items() if checked]
        warnings.warn(
            f"Selected qubits are not sorted for {problematic_cells} cells.",
            RuntimeWarning,
        )

    return rho_m_dict
