"""Post Processing - Classical Shadow - Classical Shadow
(:mod:`qurry.process.classical_shadow.classical_shadow`)

"""

from typing import Literal, Union, Optional, TypedDict, Iterable
import warnings
from itertools import combinations
import tqdm
import numpy as np

from .rho_m_core import rho_m_core
from ..availability import (
    availablility,
    default_postprocessing_backend,
    PostProcessingBackendLabel,
)


RUST_AVAILABLE = False
FAILED_RUST_IMPORT = None

BACKEND_AVAILABLE = availablility(
    "classical_shadow.classical_shadow",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(RUST_AVAILABLE, False)


class ClassicalShadowBasic(TypedDict):
    """The basic information of the classical shadow."""

    rho_m_dict: dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    """The dictionary of Rho M."""
    classical_registers_actually: list[int]
    """The list of the selected_classical_registers."""
    taking_time: float
    """The time taken for the calculation."""


class ClassicalShadowExpectation(ClassicalShadowBasic):
    """The expectation value of Rho."""

    expect_rho: np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    """The expectation value of Rho."""


def expectation_rho_core(
    rho_m_dict: dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    selected_classical_registers_sorted: list[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Calculate the expectation value of Rho.

    Args:
        rho_m_dict (dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The dictionary of Rho M.
            The dictionary of Rho M I.
        selected_classical_registers_sorted (list[int]):
            The list of the selected_classical_registers.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]: The expectation value of Rho.
    """

    expect_rho = np.zeros(
        (
            2 ** len(selected_classical_registers_sorted),
            2 ** len(selected_classical_registers_sorted),
        ),
        dtype=np.complex128,
    )
    expect_rho += np.sum(list(rho_m_dict.values()), axis=0)
    expect_rho /= len(rho_m_dict)

    return expect_rho


def expectation_rho(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    pbar: Optional[tqdm.tqdm] = None,
    multiprocess: bool = True,
) -> ClassicalShadowExpectation:
    r"""Expectation value of Rho.

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
        backend (PostProcessingBackendLabel, optional):
            The backend for the postprocessing.
            Defaults to DEFAULT_PROCESS_BACKEND.
        pbar (Optional[tqdm.tqdm], optional):
            The progress bar.
            Defaults to None.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.

    Returns:
        ClassicalShadowExpectation: The expectation value of Rho.
    """

    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    else:
        raise TypeError(
            "The selected_classical_registers should be Iterable, "
            + f"not {type(selected_classical_registers)}."
        )

    rho_m_dict, selected_classical_registers_sorted, msg, taken = rho_m_core(
        shots,
        counts,
        random_unitary_um,
        selected_classical_registers,
        backend,
        multiprocess,
    )
    if pbar is not None:
        pbar.set_description(msg)

    expect_rho = expectation_rho_core(
        rho_m_dict=rho_m_dict,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )

    return ClassicalShadowExpectation(
        expect_rho=expect_rho,
        rho_m_dict=rho_m_dict,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
    )


class ClassicalShadowPurity(ClassicalShadowBasic):
    """The expectation value of Rho."""

    purity: Union[float, np.float64]
    """The purity calculated by classical shadow."""
    entropy: Union[float, np.float64]
    """The entropy calculated by classical shadow."""


def trace_rho_square_core(
    rho_m_dict: dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    method: Literal["trace_of_matmul", "hilbert_schmidt_inner_product"] = "trace_of_matmul",
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
        rho_m_dict (dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]):
            The dictionary of Rho M.
        method (Literal["trace_of_matmul", "hilbert_schmidt_inner_product"], optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul": Use np.trace(np.matmul(rho_m1, rho_m2)) to calculate the trace.
            - "hilbert_schmidt_inner_product":
                Use np.einsum("ij,ij", rho_m1, rho_m2) to calculate the trace.
                Defaults to "trace_of_matmul".

            "hilbert_schmidt_inner_product" is inspired by Frobenius inner product
            or Hilbert-Schmidt operator
            Although it considers $Tr(A^*B)$ where A, B are matrices,
            $A^*$ is the conjugate transpose of A, which is not the $Tr(AB)$, the trace we want.
            But the implementation of Hilbert-Schmidt operator on Google Cirq,
            the quantum computing package by Google, just uses the following line:

            .. code-block:: python
                np.einsum('ij,ij', m1.conj(), m2)

            This inspired us to use

            .. code-block:: python
                np.einsum("ij,ij", rho_m1.conj(), rho_m2)
                + np.einsum("ij,ij", rho_m2.conj(), rho_m1)

            to calculate the trace. And somehow, it is the same as

            .. code-block:: python
                np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))

            Also, the einsum method is much faster than the matmul method for
            it decreases the complexity from O(n^3) to O(n^2)
            on the unused matrix elements of matrix product.

    Returns:
        np.complex128: The trace of Rho square.
    """

    num_n_u = len(rho_m_dict)
    rho_traced_sum: np.complex128 = np.complex128(0)

    rho_m_dict_combinations = combinations(rho_m_dict.items(), 2)

    if method == "hilbert_schmidt_inner_product":
        trace_array = np.array(
            [
                np.einsum("ij,ij", rho_m1.conj(), rho_m2)
                + np.einsum("ij,ij", rho_m2.conj(), rho_m1)
                for (_idx1, rho_m1), (_idx2, rho_m2) in rho_m_dict_combinations
            ]
        )
    else:
        trace_array = np.array(
            [
                np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))
                for (_idx1, rho_m1), (_idx2, rho_m2) in rho_m_dict_combinations
            ]
        )

    rho_traced_sum += trace_array.sum(dtype=np.complex128)
    rho_traced_sum /= num_n_u * (num_n_u - 1)

    assert len(trace_array) * 2 == num_n_u * (num_n_u - 1), (
        f"The number of combinations: {len(trace_array)} "
        + f"and the number of num_n_u * (num_n_u - 1): {num_n_u * (num_n_u - 1)} are different."
    )

    return rho_traced_sum


def trace_rho_square(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    method: Literal["trace_of_matmul", "hilbert_schmidt_inner_product"] = "trace_of_matmul",
    multiprocess: bool = True,
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
        backend (PostProcessingBackendLabel, optional):
            The backend for the postprocessing.
            Defaults to DEFAULT_PROCESS_BACKEND.
        method (Literal["trace_of_matmul", "hilbert_schmidt_inner_product"], optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul": Use np.trace(np.matmul(rho_m1, rho_m2)) to calculate the trace.
            - "hilbert_schmidt_inner_product":
                Use np.einsum("ij,ij", rho_m1, rho_m2) to calculate the trace.
                Defaults to "trace_of_matmul".

            "hilbert_schmidt_inner_product" is inspired by Frobenius inner product
            or Hilbert-Schmidt operator
            Although it considers $Tr(A^*B)$ where A, B are matrices,
            $A^*$ is the conjugate transpose of A, which is not the $Tr(AB)$, the trace we want.
            But the implementation of Hilbert-Schmidt operator on Google Cirq,
            the quantum computing package by Google, just uses the following line:

            .. code-block:: python
                np.einsum('ij,ij', m1.conj(), m2)

            This inspired us to use

            .. code-block:: python
                np.einsum("ij,ij", rho_m1.conj(), rho_m2)
                + np.einsum("ij,ij", rho_m2.conj(), rho_m1)

            to calculate the trace. And somehow, it is the same as

            .. code-block:: python
                np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))

            Also, the einsum method is much faster than the matmul method for
            it decreases the complexity from O(n^3) to O(n^2)
            on the unused matrix elements of matrix product.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.
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

    rho_m_dict, selected_classical_registers_sorted, msg, taken = rho_m_core(
        shots,
        counts,
        random_unitary_um,
        selected_classical_registers,
        backend,
        multiprocess,
    )
    if pbar is not None:
        pbar.set_description(msg)

    trace_rho_sum = trace_rho_square_core(rho_m_dict=rho_m_dict, method=method)
    trace_rho_sum_real = trace_rho_sum.real
    if trace_rho_sum.imag != 0:
        warnings.warn(
            "The imaginary part of the trace of Rho square is not zero. "
            + f"The imaginary part is {trace_rho_sum.imag}.",
        )
    entropy = -np.log2(trace_rho_sum_real)

    return ClassicalShadowPurity(
        purity=trace_rho_sum_real,
        entropy=entropy,
        rho_m_dict=rho_m_dict,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
    )


class ClassicalShadowComplex(ClassicalShadowBasic):
    """The expectation value of Rho and the purity calculated by classical shadow."""

    expect_rho: np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    """The expectation value of Rho."""
    purity: Union[float, np.float64]
    """The purity calculated by classical shadow."""
    entropy: Union[float, np.float64]
    """The entropy calculated by classical shadow."""


def classical_shadow_complex(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    method: Literal["trace_of_matmul", "hilbert_schmidt_inner_product"] = "trace_of_matmul",
    multiprocess: bool = True,
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
        backend (PostProcessingBackendLabel, optional):
            The backend for the postprocessing.
            Defaults to DEFAULT_PROCESS_BACKEND.
        method (Literal["trace_of_matmul", "hilbert_schmidt_inner_product"], optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul": Use np.trace(np.matmul(rho_m1, rho_m2)) to calculate the trace.
            - "hilbert_schmidt_inner_product":
                Use np.einsum("ij,ij", rho_m1, rho_m2) to calculate the trace.
                Defaults to "trace_of_matmul".

            "hilbert_schmidt_inner_product" is inspired by Frobenius inner product
            or Hilbert-Schmidt operator
            Although it considers $Tr(A^*B)$ where A, B are matrices,
            $A^*$ is the conjugate transpose of A, which is not the $Tr(AB)$, the trace we want.
            But the implementation of Hilbert-Schmidt operator on Google Cirq,
            the quantum computing package by Google, just uses the following line:

            .. code-block:: python
                np.einsum('ij,ij', m1.conj(), m2)

            This inspired us to use

            .. code-block:: python
                np.einsum("ij,ij", rho_m1.conj(), rho_m2)
                + np.einsum("ij,ij", rho_m2.conj(), rho_m1)

            to calculate the trace. And somehow, it is the same as

            .. code-block:: python
                np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))

            Also, the einsum method is much faster than the matmul method for
            it decreases the complexity from O(n^3) to O(n^2)
            on the unused matrix elements of matrix product.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.
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

    rho_m_dict, selected_classical_registers_sorted, msg, taken = rho_m_core(
        shots,
        counts,
        random_unitary_um,
        selected_classical_registers,
        backend,
        multiprocess,
    )
    if pbar is not None:
        pbar.set_description(msg)

    expect_rho = expectation_rho_core(
        rho_m_dict=rho_m_dict,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )

    trace_rho_sum = trace_rho_square_core(rho_m_dict=rho_m_dict, method=method)
    if trace_rho_sum.imag != 0:
        warnings.warn(
            "The imaginary part of the trace of Rho square is not zero. "
            + f"The imaginary part is {trace_rho_sum.imag}.",
        )
    trace_rho_sum_real = trace_rho_sum.real
    entropy = -np.log2(trace_rho_sum_real)

    return ClassicalShadowComplex(
        expect_rho=expect_rho,
        purity=trace_rho_sum_real,
        entropy=entropy,
        rho_m_dict=rho_m_dict,
        classical_registers_actually=selected_classical_registers_sorted,
        taking_time=taken,
    )
