"""Post Processing - Classical Shadow - Classical Shadow - Trace of Rho Square
(:mod:`qurry.process.classical_shadow.classical_shadow.trace`)

"""

from typing import Literal, Union, Optional, Iterable
import tqdm

from .container_kind import ClassicalShadowPurity, purity_value_kind
from ..rho_process import rho_core, RhoMethodType, DEFAULT_RHO_METHOD
from ..trace_process import all_trace_core, TraceMethodType, DEFAULT_TRACE_METHOD
from ..utils import check_random_basis_array


def trace_rho_square(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowPurity:
    """Trace of Rho square.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.

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

            - Skip calculation of trace:
                - "skip_trace": Skip the trace calculation and return NaN.

            The default method is "bitwise_py", which is the fastest option.

        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        float: The trace of Rho.
    """

    check_random_basis_array(random_basis_array, len(counts), len(next(iter(counts[0].keys()))))
    if len(counts) < 2:
        raise ValueError(
            "The method of classical shadow require at least 2 counts for the calculation. "
            + f"The number of counts is {len(counts)}."
        )

    rho_m_list, selected_classical_registers_sorted, taken = rho_core(
        shots=shots,
        counts=counts,
        random_unitary_array=random_basis_array,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    purity, entropy = all_trace_core(
        shots=shots,
        counts=counts,
        random_basis_array=random_basis_array,
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
        trace_method=trace_method,
    )

    return ClassicalShadowPurity(
        average_classical_snapshots_rho=dict(enumerate(rho_m_list)),
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        shots=shots,
        snapshots=len(rho_m_list),
        rho_method=rho_method,
        # The trace of Rho square
        purity=purity,
        entropy=entropy,
        trace_method=trace_method,
        purity_value_kind=purity_value_kind(rho_method, trace_method),
    )
