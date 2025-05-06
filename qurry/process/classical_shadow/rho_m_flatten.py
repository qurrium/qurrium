"""Post Processing - Classical Shadow - Rho M Flatten
(:mod:`qurry.process.classical_shadow.rho_m_flatten`)

"""

import time
from typing import Literal, Union
import numpy as np


from .matrix_calcution import (
    select_rho_mki_kronecker_product_2,
    ClassicalShadowPythonMethod,
    DEFAULT_PYTHON_METHOD,
)
from ..utils import (
    counts_list_under_degree_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    counts_list_vectorize_pyrust,
)


def rho_m_flatten_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    method: ClassicalShadowPythonMethod = DEFAULT_PYTHON_METHOD,
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
        method (ClassicalShadowPythonMethod, optional):
            It can be either "jax" or "numpy".
            - "jax": Use JAX to calculate the Kronecker product.
            - "numpy": Use Numpy to calculate the Kronecker product.

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
    rho_mki_kronecker_product_2 = select_rho_mki_kronecker_product_2(method=method)

    begin = time.time()

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    num_classical_register = len(selected_classical_registers_sorted)
    counts_under_degree_list = counts_list_under_degree_pyrust(
        counts,
        num_classical_register=measured_system_size,
        selected_classical_registers_sorted=selected_classical_registers_sorted,
    )
    counts_under_degree_list_vectorized = counts_list_vectorize_pyrust(counts_under_degree_list)

    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = []
    for per_um, (bit_array_as_list, value_array_as_list) in zip(
        random_unitary_um.values(), counts_under_degree_list_vectorized
    ):

        keys_int_array: np.ndarray[tuple[int, int], np.dtype[np.int32]] = np.array(
            bit_array_as_list, dtype=np.int32
        )
        nu_expanded: np.ndarray[tuple[int, int], np.dtype[np.int32]] = np.broadcast_to(
            [per_um[ci] for ci in selected_classical_registers_sorted],
            (len(bit_array_as_list), num_classical_register),
        )
        lookup_keys: np.ndarray[tuple[int, int], np.dtype[np.int32]] = (
            nu_expanded * 10 + keys_int_array
        )

        rho_m_k_weighted = np.array(
            [v * rho_mki_kronecker_product_2(kl) for kl, v in zip(lookup_keys, value_array_as_list)]
        )

        rho_m = rho_m_k_weighted.sum(axis=0)

        rho_m_list.append(rho_m / sum(value_array_as_list))

    taken = time.time() - begin

    return rho_m_list, selected_classical_registers_sorted, taken
