"""Post Processing - Classical Shadow - Classical Shadow - Mean of Rho
(:mod:`qurry.process.classical_shadow.classical_shadow.mean`)

"""

from typing import Literal, Union, Optional, Iterable
import tqdm

from .container_kind import ClassicalShadowMeanRho
from ..rho_process import rho_core, RhoMethodType, DEFAULT_RHO_METHOD, mean_rho_core
from ..utils import check_random_basis_array


def mean_of_rho(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    pbar: Optional[tqdm.tqdm] = None,
) -> ClassicalShadowMeanRho:
    r"""Calculate the mean of Rho.

    Reference:
        -   Predicting many properties of a quantum system from very few measurements -
            Huang, Hsin-Yuan and Kueng, Richard and Preskill, John
            `doi:10.1038/s41567-020-0932-7 <https://doi.org/10.1038/s41567-020-0932-7>`_

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

        pbar (Optional[tqdm.tqdm], optional):
            The progress bar. Defaults to None.

    Returns:
        ClassicalShadowMeanRho: The expectation value of Rho.
    """

    check_random_basis_array(random_basis_array, len(counts), len(next(iter(counts[0].keys()))))

    rho_m_list, selected_classical_registers_sorted, taken = rho_core(
        shots=shots,
        counts=counts,
        random_unitary_array=random_basis_array,
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
        shots=shots,
        snapshots=len(rho_m_list),
        rho_method=rho_method,
        # The mean of Rho
        mean_of_rho=expect_rho,
    )
