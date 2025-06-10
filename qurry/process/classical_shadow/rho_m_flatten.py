"""Post Processing - Classical Shadow - Rho M Flatten
(:mod:`qurry.process.classical_shadow.rho_m_flatten`)

"""

import time
from typing import Literal, Union
import numpy as np


from .matrix_calcution import rho_mki_kronecker_product_numpy_2
from ..utils import (
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
)


def rho_m_flatten_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
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
    counts_under_degree_list = counts_list_recount_pyrust(
        counts,
        num_classical_register=measured_system_size,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )
    rho_m_flatten_counts_recounting_list_vectorized = rho_m_flatten_counts_list_vectorize_pyrust(
        counts_under_degree_list, random_unitary_um, selected_classical_registers_sorted
    )

    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = []
    for bit_array_as_list, value_array_as_list in rho_m_flatten_counts_recounting_list_vectorized:

        rho_m_k_weighted = np.array(
            [
                v * rho_mki_kronecker_product_numpy_2(kl)
                for kl, v in zip(bit_array_as_list, value_array_as_list)
            ]
        )

        rho_m = rho_m_k_weighted.sum(axis=0)

        rho_m_list.append(rho_m / sum(value_array_as_list))

    taken = time.time() - begin

    return rho_m_list, selected_classical_registers_sorted, taken
