"""Post Processing - Classical Shadow - Trace-Preidction Process - Trace and Mean of Rho
(:mod:`qurry.process.classical_shadow.trace_predict_process.trace_mean_core`)

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
    JAX_AVAILABLE,
)
from ...utils import BaseMethodEnum


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


class RhoTraceMethod(BaseMethodEnum):
    """The method to use for the trace calculation of Rho square.

    - "trace_of_matmul": Use `np.trace(np.matmul(rho_m1, rho_m2))`
        to calculate the each summation item in `rho_m_list`.
    - "einsum_ij_ji": Use `np.einsum("ij,ji", rho_m1, rho_m2)`
        to calculate the each summation item in `rho_m_list`.
    - "einsum_aij_bji_to_ab_numpy": Use
        `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
        This is the fastest implementation to calculate the trace of Rho
        if JAX is not available.
    - "einsum_aij_bji_to_ab_jax": Use
        `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
        This is the fastest implementation to calculate the trace of Rho
        if JAX is available.
    """

    TRACE_OF_MATMUL = SingleTraceMethod.TRACE_OF_MATMUL.value
    """Use `np.trace(np.matmul(rho_m1, rho_m2))` to calculate 
    the each summation item in `rho_m_list`."""

    EINSUM_IJ_JI = SingleTraceMethod.EINSUM_IJ_JI.value
    """Use `np.einsum("ij,ji", rho_m1, rho_m2)` to calculate 
    the each summation item in `rho_m_list`."""

    EINSUM_AIJ_BJI_TO_AB_NUMPY = ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY.value
    """Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""

    EINSUM_AIJ_BJI_TO_AB_JAX = ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX.value
    """Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""

    @classmethod
    def get_default(cls) -> "RhoTraceMethod":
        """Get the default method.
        Returns:
            RhoTraceMethod: The default method.
        """
        return cls.EINSUM_AIJ_BJI_TO_AB_JAX if JAX_AVAILABLE else cls.EINSUM_AIJ_BJI_TO_AB_NUMPY

    def is_list_method(self) -> bool:
        """Whether it is a list method.

        Returns:
            bool: True if it is a list method, False otherwise.
        """
        return self in [self.EINSUM_AIJ_BJI_TO_AB_NUMPY, self.EINSUM_AIJ_BJI_TO_AB_JAX]

    def is_single_method(self) -> bool:
        """Whether it is a single method.

        Returns:
            bool: True if it is a single method, False otherwise.
        """
        return self in [self.TRACE_OF_MATMUL, self.EINSUM_IJ_JI]

    def to_single_enum(self) -> SingleTraceMethod:
        """Convert to SingleTraceMethod enum.

        Raises:
            ValueError: If the method is not a single method.

        Returns:
            SingleTraceMethod: The corresponding SingleTraceMethod enum.
        """
        if not self.is_single_method():
            raise ValueError(f"{self.value} is not a single method")
        return SingleTraceMethod.from_string(self.value)

    def to_list_enum(self) -> ListTraceMethod:
        """Convert to ListTraceMethod enum.

        Raises:
            ValueError: If the method is not a list method.

        Returns:
            ListTraceMethod: The corresponding ListTraceMethod enum.
        """
        if not self.is_list_method():
            raise ValueError(f"{self.value} is not a list method")
        return ListTraceMethod.from_string(self.value)


RhoTraceMethodType = Union[RhoTraceMethod, str]
"""The method to calculate the trace of Rho square.

- "trace_of_matmul":
    Use `np.trace(np.matmul(rho_m1, rho_m2))`
    to calculate the each summation item in `rho_m_list`.
- "einsum_ij_ji":
    Use `np.einsum("ij,ji", rho_m1, rho_m2)`
    to calculate the each summation item in `rho_m_list`.
- "einsum_aij_bji_to_ab_numpy":
    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho
    if JAX is not available.
- "einsum_aij_bji_to_ab_jax":
    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho.
"""

DEFAULT_RHO_TRACE_METHOD: RhoTraceMethod = RhoTraceMethod.get_default()
"""The default method for the trace of Rho square."""


def trace_rho_square_core(
    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    trace_method: RhoTraceMethod = DEFAULT_RHO_TRACE_METHOD,
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
            - "einsum_ij_ji":
                Use `np.einsum("ij,ji", rho_m1, rho_m2)`
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)`
                to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho
                if JAX is not available.
            - "einsum_aij_bji_to_ab_jax":
                Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)`
                to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho.

    Returns:
        np.complex128: The trace of Rho square.
    """
    if isinstance(trace_method, str):
        trace_method = RhoTraceMethod.from_string(trace_method)

    if trace_method.is_list_method():
        list_enum = trace_method.to_list_enum()
        rho_m_array: np.ndarray = np.array(rho_m_list)
        trace_rho_by_einsum_aij_bji_to_ab = select_all_trace_rho_by_einsum_aij_bji_to_ab(list_enum)
        return trace_rho_by_einsum_aij_bji_to_ab(rho_m_array)

    if trace_method.is_single_method():
        single_enum = trace_method.to_single_enum()
        num_n_u = len(rho_m_list)
        rho_m_list_combinations = combinations(rho_m_list, 2)

        addition_method = select_single_trace_rho_method(single_enum)
        trace_array = np.array(
            [addition_method(rho_m1_and_rho_m2) for rho_m1_and_rho_m2 in rho_m_list_combinations]
        )
        rho_traced_sum = trace_array.sum(dtype=np.complex128)
        rho_traced_sum /= num_n_u * (num_n_u - 1)

        return rho_traced_sum

    raise trace_method.value_error()
