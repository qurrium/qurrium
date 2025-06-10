"""Post Processing - Classical Shadow - Trace-Expectation Process
(:mod:`qurry.process.classical_shadow.trace_expect_process`)

This module is used to process the rho dictionary for classical shadow.
"""

from typing import Union
from itertools import combinations
import numpy as np

from .matrix_calcution import (
    select_single_trace_rho_method,
    SingleTraceRhoMethod,
    select_all_trace_rho_by_einsum_aij_bji_to_ab,
    AllTraceRhoMethod,
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


TraceRhoMethod = Union[SingleTraceRhoMethod, AllTraceRhoMethod]
"""The method to calculate the trace of Rho square.
- "trace_of_matmul":
    Use np.trace(np.matmul(rho_m1, rho_m2)) 
    to calculate the each summation item in `rho_m_list`.
- "quick_trace_of_matmul" or "einsum_ij_ji":
    Use np.einsum("ij,ji", rho_m1, rho_m2) 
    to calculate the each summation item in `rho_m_list`.
- "einsum_aij_bji_to_ab_numpy":
    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
- "einsum_aij_bji_to_ab_jax":
    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
"""


def trace_rho_square_core(
    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
) -> np.complex128:
    r"""Calculate the trace of Rho square.

    Reference:
        .. note::
            - Predicting many properties of a quantum system from very few measurements -
            Huang, Hsin-Yuan and Kueng, Richard and Preskill, John
            [doi:10.1038/s41567-020-0932-7](
                https://doi.org/10.1038/s41567-020-0932-7)

            - The randomized measurement toolbox -
            Elben, Andreas and Flammia, Steven T. and Huang, Hsin-Yuan and Kueng,
            Richard and Preskill, John and Vermersch, Benoît and Zoller, Peter
            [doi:10.1038/s42254-022-00535-2](
                https://doi.org/10.1038/s42254-022-00535-2)

        .. code-block:: bibtex
            @article{cite-key,
                abstract = {
                    Predicting the properties of complex,
                    large-scale quantum systems is essential for developing quantum technologies.
                    We present an efficient method for constructing an approximate classical
                    description of a quantum state using very few measurements of the state.
                    different properties; order
                    {\$}{\$}{\{}{$\backslash$}mathrm{\{}log{\}}{\}}{$\backslash$},(M){\$}{\$}
                    measurements suffice to accurately predict M different functions of the state
                    with high success probability. The number of measurements is independent of
                    the system size and saturates information-theoretic lower bounds. Moreover,
                    target properties to predict can be
                    selected after the measurements are completed.
                    We support our theoretical findings with extensive numerical experiments.
                    We apply classical shadows to predict quantum fidelities,
                    entanglement entropies, two-point correlation functions,
                    expectation values of local observables and the energy variance of
                    many-body local Hamiltonians.
                    The numerical results highlight the advantages of classical shadows relative to
                    previously known methods.},
                author = {Huang, Hsin-Yuan and Kueng, Richard and Preskill, John},
                date = {2020/10/01},
                date-added = {2024-12-03 15:00:55 +0800},
                date-modified = {2024-12-03 15:00:55 +0800},
                doi = {10.1038/s41567-020-0932-7},
                id = {Huang2020},
                isbn = {1745-2481},
                journal = {Nature Physics},
                number = {10},
                pages = {1050--1057},
                title = {Predicting many properties of a quantum system from very few measurements},
                url = {https://doi.org/10.1038/s41567-020-0932-7},
                volume = {16},
                year = {2020},
                bdsk-url-1 = {https://doi.org/10.1038/s41567-020-0932-7}
            }

            @article{cite-key,
                abstract = {
                    Programmable quantum simulators and quantum computers are opening unprecedented
                    opportunities for exploring and exploiting the properties of highly entangled
                    complex quantum systems. The complexity of large quantum systems is the source
                    of computational power but also makes them difficult to control precisely or
                    characterize accurately using measured classical data. We review protocols
                    for probing the properties of complex many-qubit systems using measurement
                    schemes that are practical using today's quantum platforms. In these protocols,
                    a quantum state is repeatedly prepared and measured in a randomly chosen basis;
                    then a classical computer processes the measurement outcomes to estimate the
                    desired property. The randomization of the measurement procedure has distinct
                    advantages. For example, a single data set can be used multiple times to pursue
                    a variety of applications, and imperfections in the measurements are mapped to
                    a simplified noise model that can more
                    easily be mitigated. We discuss a range of
                    cases that have already been realized in quantum devices, including Hamiltonian
                    simulation tasks, probes of quantum chaos, measurements of non-local order
                    parameters, and comparison of quantum states produced in distantly separated
                    laboratories. By providing a workable method for translating a complex quantum
                    state into a succinct classical representation that preserves a rich variety of
                    relevant physical properties, the randomized measurement toolbox strengthens our
                    ability to grasp and control the quantum world.},
                author = {
                    Elben, Andreas and Flammia, Steven T. and Huang, Hsin-Yuan and Kueng,
                    Richard and Preskill, John and Vermersch, Beno{\^\i}t and Zoller, Peter},
                date = {2023/01/01},
                date-added = {2024-12-03 15:06:15 +0800},
                date-modified = {2024-12-03 15:06:15 +0800},
                doi = {10.1038/s42254-022-00535-2},
                id = {Elben2023},
                isbn = {2522-5820},
                journal = {Nature Reviews Physics},
                number = {1},
                pages = {9--24},
                title = {The randomized measurement toolbox},
                url = {https://doi.org/10.1038/s42254-022-00535-2},
                volume = {5},
                year = {2023},
                bdsk-url-1 = {https://doi.org/10.1038/s42254-022-00535-2}
            }

    Args:
        rho_m_list (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The dictionary of Rho M.
        trace_method (TraceMethod , optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul":
                Use np.trace(np.matmul(rho_m1, rho_m2))
                to calculate the each summation item in `rho_m_list`.
            - "quick_trace_of_matmul" or "einsum_ij_ji":
                Use np.einsum("ij,ji", rho_m1, rho_m2)
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.

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
