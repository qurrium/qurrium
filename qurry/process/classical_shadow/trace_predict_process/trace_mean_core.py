"""Post Processing - Classical Shadow - Trace-Preidction Process - Trace and Mean of Rho
(:mod:`qurry.process.classical_shadow.trace_predcit_process.trace_mean_core`)

This module is used to process the rho dictionary for classical shadow.
"""

from typing import Union
from itertools import combinations
import numpy as np

from .matrix_calcution import (
    select_single_trace_rho_method,
    SingleTraceMethod,
    select_all_trace_rho_by_einsum_aij_bji_to_ab,
    ListTraceMethod,
    DEFAULT_ALL_TRACE_RHO_METHOD,
)


def mean_rho_core(
    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    selected_classical_registers_sorted: list[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Calculate the expectation value of Rho.

    Args:
        rho_m_list (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The dictionary of Rho M.
            The dictionary of Rho M I.
        selected_classical_registers_sorted (list[int]):
            The list of the selected_classical_registers.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]: The expectation value of Rho.
    """

    expect_rho: np.ndarray[tuple[int, int], np.dtype[np.complex128]] = np.sum(
        rho_m_list, axis=0, dtype=np.complex128
    )  # type: ignore
    assert expect_rho.shape == (2 ** len(selected_classical_registers_sorted),) * 2, (
        f"The shape of expect_rho: {expect_rho.shape} "
        + f"and the shape of rho_m_list: {rho_m_list[0].shape} are different."
    )
    expect_rho /= len(rho_m_list)

    return expect_rho


TraceMethod = Union[SingleTraceMethod, ListTraceMethod]
"""The method to calculate the trace of Rho square.

- "trace_of_matmul":
    Use `np.trace(np.matmul(rho_m1, rho_m2))`
    to calculate the each summation item in `rho_m_list`.
- "quick_trace_of_matmul" or "einsum_ij_ji":
    Use `np.einsum("ij,ji", rho_m1, rho_m2)`
    to calculate the each summation item in `rho_m_list`.
- "einsum_aij_bji_to_ab_numpy":
    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
- "einsum_aij_bji_to_ab_jax":
    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
"""


def trace_rho_square_core(
    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    trace_method: TraceMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
) -> np.complex128:
    r"""Calculate the trace of Rho square.

    Args:
        rho_m_list (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The dictionary of Rho M.
        trace_method (TraceMethod , optional):
            The method to calculate the trace of Rho square.

            - "trace_of_matmul":
                Use `np.trace(np.matmul(rho_m1, rho_m2))`
                to calculate the each summation item in `rho_m_list`.
            - "quick_trace_of_matmul" or "einsum_ij_ji":
                Use `np.einsum("ij,ji", rho_m1, rho_m2)`
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.

    Returns:
        np.complex128: The trace of Rho square.
    """

    if trace_method in ["einsum_aij_bji_to_ab_numpy", "einsum_aij_bji_to_ab_jax"]:
        rho_m_array: np.ndarray = np.array(rho_m_list)
        trace_rho_by_einsum_aij_bji_to_ab = select_all_trace_rho_by_einsum_aij_bji_to_ab(
            trace_method
        )
        return trace_rho_by_einsum_aij_bji_to_ab(rho_m_array)

    num_n_u = len(rho_m_list)
    rho_m_list_combinations = combinations(rho_m_list, 2)

    addition_method = select_single_trace_rho_method(trace_method)
    trace_array = np.array(
        [addition_method(rho_m1_and_rho_m2) for rho_m1_and_rho_m2 in rho_m_list_combinations]
    )
    rho_traced_sum = trace_array.sum(dtype=np.complex128)
    rho_traced_sum /= num_n_u * (num_n_u - 1)

    return rho_traced_sum
