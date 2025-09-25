"""Post Processing - Classical Shadow - All Trace Process
(:mod:`qurry.process.classical_shadow.all_trace_process`)

"""

from typing import Literal, Union
import warnings
import numpy as np

from .rho_process import RhoMethod, RhoMethodType, DEFAULT_RHO_METHOD
from .trace_predict_process import trace_rho_square_core, RhoTraceMethod
from .nomatop_process import trace_nomatop_core, NonMatOpTraceMethod
from .container import PurityValueKind
from ..utils import NUMERICAL_ERROR_TOLERANCE, BaseMethodEnum


class TraceMethod(BaseMethodEnum):
    """The method to calculate the trace of rho.

    - Matrix operation methods:
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

    For the matrix operation methods, it will require rho has been calculated first.

    - Non-matrix operation methods:
        - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
        - "nomatmul_trace_rust": Use Rust implementation via PyO3.
        - "bitwise_py": Use pure Python bitwise implementation.

    For the non-matrix operation methods, it will directly calculate the trace from
    the counts and random basis.

    The default method is "bitwise_py", which is the fastest option.
    """

    # Matrix operation methods
    TRACE_OF_MATMUL = RhoTraceMethod.TRACE_OF_MATMUL.value
    """Use `np.trace(np.matmul(rho_m1, rho_m2))` to calculate
    the each summation item in `rho_m_list`."""
    EINSUM_IJ_JI = RhoTraceMethod.EINSUM_IJ_JI.value
    """Use `np.einsum("ij,ji", rho_m1, rho_m2)` to calculate
    the each summation item in `rho_m_list`."""
    EINSUM_AIJ_BJI_TO_AB_NUMPY = RhoTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY.value
    """Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""
    EINSUM_AIJ_BJI_TO_AB_JAX = RhoTraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX.value
    """Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""

    # Non-matrix operation methods
    NOMATMUL_TRACE_PY = NonMatOpTraceMethod.NOMATMUL_TRACE_PY.value
    """Use pure Python implementation without multiprocessing."""
    NOMATMUL_TRACE_RUST = NonMatOpTraceMethod.NOMATMUL_TRACE_RUST.value
    """Use Rust implementation via PyO3."""
    BITWISE_PY = NonMatOpTraceMethod.BITWISE_PY.value
    """Use pure Python bitwise implementation."""

    @classmethod
    def get_default(cls) -> "TraceMethod":
        """Get the default method.
        Returns:
            TraceMethod: The default method.
        """
        return cls.BITWISE_PY

    def is_bitwise_method(self) -> bool:
        """Whether it is a bitwise method.

        Returns:
            bool: True if it is a bitwise method, False otherwise.
        """
        return self in [self.BITWISE_PY]

    @classmethod
    def get_all_bitwise_methods(cls) -> list[str]:
        """Get a list of all avaialble bitwise methods.

        Returns:
            list[str]: A list of avaialble bitwise methods.
        """
        return [cls.BITWISE_PY.value]

    def is_singleshots_method(self) -> bool:
        """Whether it is a singleshots method.

        Returns:
            bool: True if it is a singleshots method, False otherwise.
        """
        return self in [self.NOMATMUL_TRACE_PY, self.NOMATMUL_TRACE_RUST]

    @classmethod
    def get_single_methods(cls) -> list[str]:
        """Get a list of all avaialble non-matrix operation methods.

        Returns:
            list[str]: A list of avaialble non-matrix operation methods.
        """
        return [cls.NOMATMUL_TRACE_PY.value, cls.NOMATMUL_TRACE_RUST.value]

    def is_nomatop_method(self) -> bool:
        """Whether it is a nomatmul method.

        Returns:
            bool: True if it is a nomatmul method, False otherwise.
        """
        return self.is_bitwise_method() or self.is_singleshots_method()

    @classmethod
    def get_nomatop_methods(cls) -> list[str]:
        """Get a list of all avaialble nomatmul methods.

        Returns:
            list[str]: A list of avaialble nomatmul methods.
        """
        return cls.get_single_methods() + cls.get_all_bitwise_methods()

    def to_nomatop_enum(self) -> NonMatOpTraceMethod:
        """Convert to NonMatOpTraceMethod enum.

        Raises:
            ValueError: If the method is not a nomatmul method.

        Returns:
            NonMatOpTraceMethod: The corresponding NonMatOpTraceMethod enum.
        """
        if not self.is_nomatop_method():
            raise ValueError(f"{self.value} is not a nomatmul method")
        return NonMatOpTraceMethod.from_string(self.value)

    def is_matrixop_method(self) -> bool:
        """Whether it is a matrix operation method.

        Returns:
            bool: True if it is a matrix operation method, False otherwise.
        """
        return self in [
            self.TRACE_OF_MATMUL,
            self.EINSUM_IJ_JI,
            self.EINSUM_AIJ_BJI_TO_AB_NUMPY,
            self.EINSUM_AIJ_BJI_TO_AB_JAX,
        ]

    @classmethod
    def get_all_matrixop_methods(cls) -> list[str]:
        """Get a list of all avaialble matrix operation methods.

        Returns:
            list[str]: A list of avaialble matrix operation methods.
        """
        return [
            cls.TRACE_OF_MATMUL.value,
            cls.EINSUM_IJ_JI.value,
            cls.EINSUM_AIJ_BJI_TO_AB_NUMPY.value,
            cls.EINSUM_AIJ_BJI_TO_AB_JAX.value,
        ]

    def to_matrixop_enum(self) -> RhoTraceMethod:
        """Convert to RhoTraceMethod enum.

        Raises:
            ValueError: If the method is not a matrix operation method.

        Returns:
            RhoTraceMethod: The corresponding RhoTraceMethod enum.
        """
        if not self.is_matrixop_method():
            raise ValueError(f"{self.value} is not a matrix operation method")
        return RhoTraceMethod.from_string(self.value)


TraceMethodType = Union[TraceMethod, str]
"""The method to calculate the trace of rho.

- Matrix operation methods:
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

For the matrix operation methods, it will require rho has been calculated first.

- Non-matrix operation methods:
    - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
    - "nomatmul_trace_rust": Use Rust implementation via PyO3.
    - "bitwise_py": Use pure Python bitwise implementation.

For the non-matrix operation methods, it will directly calculate the trace from
the counts and random basis.

The default method is "bitwise_py", which is the fastest option.
"""


DEFAULT_TRACE_METHOD: TraceMethod = TraceMethod.get_default()
"""The default method for the trace calculation of rho."""


def all_trace_core(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Union[Literal[0, 1, 2], int]]],
    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    selected_classical_registers_sorted: list[int],
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
) -> tuple[Union[float, np.float64], Union[float, np.float64]]:
    """Calculate the trace by all given methods.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.
        rho_m_list (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The list of Rho M.
        selected_classical_registers_sorted (list[int]):
            The **sorted** list of the index of the selected classical registers.

        rho_method (RhoMethodType, optional):
            It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
            "single_shots_proto", "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots_proto": Use Numpy to calculate the rho_m.
            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots_proto": Use Numpy to calculate the rho_m
                with converted single shot counts.
            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        trace_method (TraceMethodType, optional):
            The method to calculate the trace of rho.

            - Matrix operation methods:
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
            For the matrix operation methods, it will require rho has been calculated first.

            - Non-matrix operation methods:
                - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
                - "nomatmul_trace_rust": Use Rust implementation via PyO3.
                - "bitwise_py": Use pure Python bitwise implementation.
            For the non-matrix operation methods, it will directly calculate the trace from
            the counts and random basis.

            The default method is "bitwise_py", which is the fastest option.

    Returns:
        tuple[Union[float, np.float64], Union[float, np.float64]]:
            The purity and the second Renyi entropy.
    """

    if isinstance(trace_method, str):
        trace_method = TraceMethod.from_string(trace_method)

    if trace_method.is_nomatop_method():
        purity = trace_nomatop_core(
            shots=shots,
            counts=counts,
            random_unitary_array=random_basis_array,
            selected_classical_registers=selected_classical_registers_sorted,
            trace_method=trace_method.to_nomatop_enum(),
        )
    else:
        trace_rho_sum = trace_rho_square_core(
            rho_m_list=rho_m_list, trace_method=trace_method.to_matrixop_enum()
        )
        if np.abs(trace_rho_sum.imag) > NUMERICAL_ERROR_TOLERANCE:
            warnings.warn(
                "The imaginary part of the trace of Rho square is not zero. "
                f"The imaginary part is {trace_rho_sum.imag}. method: {trace_method}, {rho_method}",
                RuntimeWarning,
            )
        purity = trace_rho_sum.real

    entropy = -np.log2(purity)

    return purity, entropy


def purity_value_kind(rho_method: RhoMethodType, trace_method: TraceMethodType) -> PurityValueKind:
    """Get the kind of purity value calculation.

    Args:
        rho_method (RhoMethodType, optional):
            It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
            "single_shots_proto", "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots_proto": Use Numpy to calculate the rho_m.
            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots_proto": Use Numpy to calculate the rho_m
                with converted single shot counts.
            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        trace_method (TraceMethodType, optional):
            The method to calculate the trace of rho.

            - Matrix operation methods:
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

            For the matrix operation methods, it will require rho has been calculated first.

            - Non-matrix operation methods:
                - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
                - "nomatmul_trace_rust": Use Rust implementation via PyO3.
                - "bitwise_py": Use pure Python bitwise implementation.

            For the non-matrix operation methods, it will directly calculate the trace from
            the counts and random basis.

            The default method is "bitwise_py", which is the fastest option.

    Returns:
        PurityValueKind: The kind of purity value calculation.
    """
    if isinstance(rho_method, str):
        rho_method = RhoMethod.from_string(rho_method)
    if isinstance(trace_method, str):
        trace_method = TraceMethod.from_string(trace_method)

    if trace_method.is_bitwise_method():
        return "bitwise"
    if not trace_method.is_nomatop_method() and rho_method.is_multi_method():
        return "multi_shots"
    return "single_shots"
