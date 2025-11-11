"""Post Processing - Classical Shadow - Classical Shadow - Complex of All Variables
(:mod:`qurry.process.classical_shadow.classical_shadow.complex`)

"""

from typing import Literal, Union, Optional, Iterable
import tqdm
import numpy as np

from .container_kind import ClassicalShadowComplex, purity_value_kind
from ..rho_process import (
    rho_core,
    RhoMethodType,
    DEFAULT_RHO_METHOD,
    ShadowBasisType,
    DEFAULT_SHADOW_BASIS,
    mean_rho_core,
)
from ..trace_process import all_trace_core, TraceMethodType, DEFAULT_TRACE_METHOD
from ..prediction_process import prediction_algorithm
from ..matrix_calculation import ListTraceMethodType, DEFAULT_LIST_TRACE_METHOD
from ..utils import check_random_basis_array


def classical_shadow_complex(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # estimation of given operators
    given_operators: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    accuracy_prob_comp_delta: float = 0.01,
    max_shadow_norm: Optional[float] = None,
    # other config
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
    trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
    estimate_trace_method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowComplex:
    r"""Calculate the expectation value of Rho and the purity by classical shadow.

    Reference:
        -   Predicting many properties of a quantum system from very few measurements -
            Huang, Hsin-Yuan and Kueng, Richard and Preskill, John
            `doi:10.1038/s41567-020-0932-7 <https://doi.org/10.1038/s41567-020-0932-7>`_

        -   The randomized measurement toolbox -
            Elben, Andreas and Flammia, Steven T. and Huang, Hsin-Yuan and Kueng,
            Richard and Preskill, John and Vermersch, Benoît and Zoller, Peter
            `doi:10.1038/s42254-022-00535-2 <https://doi.org/10.1038/s42254-022-00535-2>`_

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
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.

        given_operators (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The list of the operators to estimate.
        accuracy_prob_comp_delta (float, optional):
            The accuracy probability component delta. Defaults to 0.01.
        max_shadow_norm (Optional[float], optional):
            The maximum shadow norm. Defaults to None.
            If it is None, it will be calculated by the largest shadow norm upper bound.
            If it is not None, it must be a positive float number.
            It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

        rho_method (RhoMethodType, optional):
            It can be either "multi_shots", "multi_shots_vectorized",
            "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        shadow_basis (ShadowBasisType, optional):
            The shadow basis to use. Defaults to :data:`DEFAULT_SHADOW_BASIS`.

            Here are the built-in basis sets:
            - `RX_RY_RZ`:
                Uses :math:`R_X(\frac{\pi}{2})`, :math:`R_Y(-\frac{\pi}{2})`, and :math:`R_Z(0)` gates.
            - `H_H-Sdg_I`:
                Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`, and Identity gates.
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
        estimate_trace_method (ListTraceMethodType, optional):
            The method to use for the calculation.

            - "einsum_aij_bji_to_ab_numpy":
                Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho
                if JAX is not available.
            - "einsum_aij_bji_to_ab_jax":
                Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho.

            Defaults to DEFAULT_LIST_TRACE_METHOD.

        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        ClassicalShadowComplex:
            The expectation value of Rho and the purity calculated by classical shadow.
    """

    check_random_basis_array(random_basis_array, len(counts), len(next(iter(counts[0].keys()))))
    if len(counts) < 2:
        raise ValueError(
            "The method of classical shadow require at least 2 counts for the calculation. "
            + f"The number of counts is {len(counts)}."
        )
    kind_of_purity = purity_value_kind(rho_method, trace_method)

    rho_m_list, selected_classical_registers_sorted, shadow_basis_obj, taken = rho_core(
        shots=shots,
        counts=counts,
        random_unitary_array=random_basis_array,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
        shadow_basis=shadow_basis,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    expect_rho = mean_rho_core(
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )

    purity, entropy = all_trace_core(
        shots=shots,
        counts=counts,
        random_basis_array=random_basis_array,
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
        trace_method=trace_method,
    )

    average_classical_snapshots_rho = dict(enumerate(rho_m_list))

    all_prediction_results = prediction_algorithm(
        classical_snapshots_rho=average_classical_snapshots_rho,
        given_operators=given_operators,
        accuracy_prob_comp_delta=accuracy_prob_comp_delta,
        max_shadow_norm=max_shadow_norm,
        estimate_trace_method=estimate_trace_method,
    )
    return ClassicalShadowComplex(
        average_classical_snapshots_rho=average_classical_snapshots_rho,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        shots=shots,
        snapshots=len(rho_m_list),
        rho_method=rho_method,
        random_basis_data=shadow_basis_obj.export(),
        # The mean of Rho
        mean_of_rho=expect_rho,
        # The trace of Rho square
        purity=purity,
        entropy=entropy,
        purity_value_kind=kind_of_purity,
        trace_method=trace_method,
        # esitimation of given operators
        **all_prediction_results,
    )
