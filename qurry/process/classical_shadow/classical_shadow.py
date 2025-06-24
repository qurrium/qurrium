"""Post Processing - Classical Shadow - Classical Shadow
(:mod:`qurry.process.classical_shadow.classical_shadow`)

"""

from typing import Literal, Union, Optional, Iterable
import warnings
import tqdm
import numpy as np

from .rho_m_core import rho_m_core, RhoMCoreMethod
from .trace_expect_process import (
    mean_rho_core,
    trace_rho_square_core,
    DEFAULT_ALL_TRACE_RHO_METHOD,
    AllTraceRhoMethod,
    TraceRhoMethod,
)
from .expectation_process import prediction_algorithm
from .container import (
    ClassicalShadowMeanRho,
    ClassicalShadowEstimation,
    ClassicalShadowPurity,
    ClassicalShadowComplex,
)
from ..utils import NUMERICAL_ERROR_TOLERANCE


def mean_of_rho(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # other config
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowMeanRho:
    r"""Calculate the mean of Rho.

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
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        backend (PostProcessingBackendLabel, optional):
            The backend for the postprocessing.
            Defaults to DEFAULT_PROCESS_BACKEND.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        ClassicalShadowMeanRho: The expectation value of Rho.
    """

    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    else:
        raise TypeError(
            "The selected_classical_registers should be Iterable, "
            + f"not {type(selected_classical_registers)}."
        )

    rho_m_list, selected_classical_registers_sorted, taken = rho_m_core(
        shots=shots,
        counts=counts,
        random_unitary_um=random_unitary_um,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    expect_rho = mean_rho_core(
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )

    return ClassicalShadowMeanRho(
        average_classical_snapshots_rho=dict(enumerate(rho_m_list)),
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        # The mean of Rho
        mean_of_rho=expect_rho,
    )


def trace_rho_square(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # other config
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowPurity:
    """Trace of Rho square.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.

        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        trace_method (TraceRhoMethod, optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul":
                Use np.trace(np.matmul(rho_m1, rho_m2))
                to calculate the each summation item in `rho_m_list`.
            - "quick_trace_of_matmul" or "einsum_ij_ji":
                Use np.einsum("ij,ji", rho_m1, rho_m2)
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        float: The trace of Rho.
    """

    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    else:
        raise TypeError(
            "The selected_classical_registers should be Iterable, "
            + f"not {type(selected_classical_registers)}."
        )

    if len(counts) < 2:
        raise ValueError(
            "The method of classical shadow require at least 2 counts for the calculation. "
            + f"The number of counts is {len(counts)}."
        )

    rho_m_list, selected_classical_registers_sorted, taken = rho_m_core(
        shots=shots,
        counts=counts,
        random_unitary_um=random_unitary_um,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    trace_rho_sum = trace_rho_square_core(rho_m_list=rho_m_list, trace_method=trace_method)
    trace_rho_sum_real = trace_rho_sum.real
    if np.abs(trace_rho_sum.imag) > NUMERICAL_ERROR_TOLERANCE:
        warnings.warn(
            "The imaginary part of the trace of Rho square is not zero. "
            f"The imaginary part is {trace_rho_sum.imag}. method: {trace_method}, {rho_method}",
            RuntimeWarning,
        )
    entropy = -np.log2(trace_rho_sum_real)

    return ClassicalShadowPurity(
        average_classical_snapshots_rho=dict(enumerate(rho_m_list)),
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        # The trace of Rho square
        purity=trace_rho_sum_real,
        entropy=entropy,
    )


def esitimation_of_given_operators(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # estimation of given operators
    given_operators: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    accuracy_prob_comp_delta: float = 0.01,
    max_shadow_norm: Optional[float] = None,
    # other config
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowEstimation:
    r"""Calculate the expectation value of given operators.

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

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.

        given_operators (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The list of the operators to estimate.
        accuracy_prob_comp_delta (float, optional):
            The accuracy probability component delta. Defaults to 0.01.
        max_shadow_norm (Optional[float], optional):
            The maximum shadow norm. Defaults to None.
            If it is None, it will be calculated by the largest shadow norm upper bound.
            If it is not None, it must be a positive float number.
            It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        estimate_trace_method (AllTraceRhoMethod, optional):
            The method to calculate the trace for searching esitmator.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        ClassicalShadowEstimation: The estimation of the given operators.
    """
    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    else:
        raise TypeError(
            "The selected_classical_registers should be Iterable, "
            + f"not {type(selected_classical_registers)}."
        )

    rho_m_list, selected_classical_registers_sorted, taken = rho_m_core(
        shots=shots,
        counts=counts,
        random_unitary_um=random_unitary_um,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")
    average_classical_snapshots_rho = dict(enumerate(rho_m_list))

    (
        estimate_of_given_operators,
        corresponding_rhos,
        actual_accuracy_prob_comp_delta,
        num_of_estimators,
        accuracy_predict_epsilon,
        max_shadow_norm,
        epsilon_upperbound,
        shadow_norm_upperbound,
    ) = prediction_algorithm(
        classical_snapshots_rho=average_classical_snapshots_rho,
        given_operators=given_operators,
        accuracy_prob_comp_delta=accuracy_prob_comp_delta,
        max_shadow_norm=max_shadow_norm,
        trace_method=estimate_trace_method,
    )

    return ClassicalShadowEstimation(
        average_classical_snapshots_rho=average_classical_snapshots_rho,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        # esitimation of given operators
        estimate_of_given_operators=estimate_of_given_operators,
        corresponding_rhos=corresponding_rhos,
        accuracy_prob_comp_delta=actual_accuracy_prob_comp_delta,
        num_of_estimators_k=num_of_estimators,
        accuracy_predict_epsilon=accuracy_predict_epsilon,
        maximum_shadow_norm=max_shadow_norm,
        epsilon_upperbound=epsilon_upperbound,
        shadow_norm_upperbound=shadow_norm_upperbound,
    )


def classical_shadow_complex(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # estimation of given operators
    given_operators: Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]] = None,
    accuracy_prob_comp_delta: float = 0.01,
    max_shadow_norm: Optional[float] = None,
    # other config
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowComplex:
    r"""Calculate the expectation value of Rho and the purity by classical shadow.

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
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.

        given_operators (list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The list of the operators to estimate. Defaults to None.
        accuracy_prob_comp_delta (float, optional):
            The accuracy probability component delta. Defaults to 0.01.
        max_shadow_norm (Optional[float], optional):
            The maximum shadow norm. Defaults to None.
            If it is None, it will be calculated by the largest shadow norm upper bound.
            If it is not None, it must be a positive float number.
            It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        trace_method (TraceRhoMethod, optional):
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
        estimate_trace_method (AllTraceRhoMethod, optional):
            The method to calculate the trace for searching esitmator.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        ClassicalShadowComplex:
            The expectation value of Rho and the purity calculated by classical shadow.
    """

    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    else:
        raise TypeError(
            "The selected_classical_registers should be Iterable, "
            + f"not {type(selected_classical_registers)}."
        )

    rho_m_list, selected_classical_registers_sorted, taken = rho_m_core(
        shots=shots,
        counts=counts,
        random_unitary_um=random_unitary_um,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
    )
    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    expect_rho = mean_rho_core(
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )

    trace_rho_sum = trace_rho_square_core(rho_m_list=rho_m_list, trace_method=trace_method)
    if np.abs(trace_rho_sum.imag) > NUMERICAL_ERROR_TOLERANCE:
        warnings.warn(
            "The imaginary part of the trace of Rho square is not zero, "
            + f"error larger than the tolerance of {NUMERICAL_ERROR_TOLERANCE}. "
            + f"The imaginary part is {trace_rho_sum.imag}. method: {trace_method}, {rho_method}.",
            RuntimeWarning,
        )
    trace_rho_sum_real = trace_rho_sum.real
    entropy = -np.log2(trace_rho_sum_real)

    average_classical_snapshots_rho = dict(enumerate(rho_m_list))

    if given_operators is None or len(given_operators) == 0:
        return ClassicalShadowComplex(
            average_classical_snapshots_rho=average_classical_snapshots_rho,
            classical_registers_actually=selected_classical_registers_sorted,
            taking_time=taken,
            # The mean of Rho
            mean_of_rho=expect_rho,
            # The trace of Rho square
            purity=trace_rho_sum_real,
            entropy=entropy,
            # esitimation of given operators
            estimate_of_given_operators=[],
            corresponding_rhos=[],
            accuracy_prob_comp_delta=np.nan,
            num_of_estimators_k=0,
            accuracy_predict_epsilon=np.nan,
            maximum_shadow_norm=np.nan,
            epsilon_upperbound=np.nan,
            shadow_norm_upperbound=np.nan,
        )

    (
        estimate_of_given_operators,
        corresponding_rhos,
        actual_accuracy_prob_comp_delta,
        num_of_estimators,
        accuracy_predict_epsilon,
        max_shadow_norm,
        epsilon_upperbound,
        shadow_norm_upperbound,
    ) = prediction_algorithm(
        classical_snapshots_rho=average_classical_snapshots_rho,
        given_operators=given_operators,
        accuracy_prob_comp_delta=accuracy_prob_comp_delta,
        max_shadow_norm=max_shadow_norm,
        trace_method=estimate_trace_method,
    )
    return ClassicalShadowComplex(
        average_classical_snapshots_rho=average_classical_snapshots_rho,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
        # The mean of Rho
        mean_of_rho=expect_rho,
        # The trace of Rho square
        purity=trace_rho_sum_real,
        entropy=entropy,
        # esitimation of given operators
        estimate_of_given_operators=estimate_of_given_operators,
        corresponding_rhos=corresponding_rhos,
        accuracy_prob_comp_delta=actual_accuracy_prob_comp_delta,
        num_of_estimators_k=num_of_estimators,
        accuracy_predict_epsilon=accuracy_predict_epsilon,
        maximum_shadow_norm=max_shadow_norm,
        epsilon_upperbound=epsilon_upperbound,
        shadow_norm_upperbound=shadow_norm_upperbound,
    )
