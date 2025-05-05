"""Post Processing - Classical Shadow - Rho M Flatten
(:mod:`qurry.process.classical_shadow.rho_m_flatten`)

"""

import time
from typing import Literal, Union
from multiprocessing import get_context
import numpy as np


from .matrix_calcution import (
    select_process_single_count,
    PostProcessingBackendClassicalShadow,
    DEFAULT_PROCESS_BACKEND_CLASSICAL_SHADOW,
)
from ..utils import counts_list_under_degree_pyrust, shot_counts_selected_clreg_checker_pyrust
from ...tools import DEFAULT_POOL_SIZE


def rho_m_flatten_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    backend: PostProcessingBackendClassicalShadow = DEFAULT_PROCESS_BACKEND_CLASSICAL_SHADOW,
    multiprocess: bool = True,
) -> tuple[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]], list[int], float]:
    """Rho M Cell Core calculation and directly return :cls:`ClassicalShadowComplex`.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.
        method (RhoMKCellMethod, optional):
            The method to use for the calculation. Defaults to "Python_precomputed".
            It can be either "Python" or "Python_precomputed".
        backend (PostProcessingBackendClassicalShadow, optional):
            It can be either "jax" or "numpy".
            - "jax": Use JAX to calculate the Kronecker product.
            - "numpy": Use Numpy to calculate the Kronecker product.
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The list of rho_m, the sorted list of the selected qubits, and calculation time.
    """

    measured_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )

    begin = time.time()

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    counts_under_degree_list = counts_list_under_degree_pyrust(
        counts,
        num_classical_register=measured_system_size,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )
    random_unitary_ids_array = np.fromiter(
        (list(v.values()) for v in random_unitary_um.values()),
        dtype=np.dtype((int, measured_system_size)),
    )
    random_unitary_ids_array_under_degree = random_unitary_ids_array[
        :, selected_classical_registers_sorted
    ]

    process_single_count = select_process_single_count(backend=backend)

    if multiprocess:
        pool = get_context("spawn").Pool(DEFAULT_POOL_SIZE)
        with pool as p:
            rho_m_list = p.starmap(
                process_single_count,
                list(zip(random_unitary_ids_array_under_degree, counts_under_degree_list)),
            )
    else:
        rho_m_list = [
            process_single_count(nu_dir_array, single_counts)
            for nu_dir_array, single_counts in zip(
                random_unitary_ids_array_under_degree, counts_under_degree_list
            )
        ]

    taken = time.time() - begin

    return rho_m_list, selected_classical_registers_sorted, taken
