"""Post Processing - Classical Shadow - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_m_core`)

"""

import time
import warnings
from typing import Literal, Union, Iterable, Optional
import numpy as np

from .matrix_calcution import JAX_AVAILABLE, FAILED_JAX_IMPORT, rho_mki_kronecker_product_numpy_2
from .rho_mk_cell import rho_mk_cell_py, rho_mk_cell_py_precomputed, RhoMKCellMethod
from .spreadout import spreadout
from .utils import check_random_basis
from ..utils import (
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
)
from ..availability import availablility

RUST_AVAILABLE = False
FAILED_RUST_IMPORT = None


BACKEND_AVAILABLE = availablility(
    "classical_shadow.rho_m_core",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
        ("numpy", True, None),
        ("JAX", JAX_AVAILABLE, FAILED_JAX_IMPORT),
    ],
)


def rho_m_flatten_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    convert_to_single_shot: bool = False,
) -> tuple[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]], list[int], float]:
    """Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        convert_to_single_shot (bool, optional):
            Whether to convert the counts and the random basis from multiple shots
            to single shot per snapshot for classical shadow post-processing.
            Default to False.

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
    check_random_basis(random_unitary_um, selected_classical_registers)
    if convert_to_single_shot:
        counts, random_unitary_um = spreadout(shots, counts, random_unitary_um)

    begin = time.time()

    selected_clregs_sorted = sorted(selected_classical_registers, reverse=True)

    counts_under_degree_list = counts_list_recount_pyrust(
        counts,
        num_classical_register=measured_system_size,
        selected_classical_registers_sorted=selected_clregs_sorted,
    )
    flatten_recount_list_vectorized = rho_m_flatten_counts_list_vectorize_pyrust(
        counts_under_degree_list, random_unitary_um, selected_clregs_sorted
    )

    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = [
        np.array(
            [
                v * rho_mki_kronecker_product_numpy_2(kl)
                for kl, v in zip(bit_array_as_list, value_array_as_list)
            ]
        ).sum(axis=0)
        / sum(value_array_as_list)
        for bit_array_as_list, value_array_as_list in flatten_recount_list_vectorized
    ]  # type: ignore

    taken = time.time() - begin

    return rho_m_list, selected_clregs_sorted, taken


def rho_m_core_py(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    convert_to_single_shot: bool = False,
    rho_method: RhoMKCellMethod = "numpy_precomputed",
) -> tuple[
    list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    list[int],
    float,
]:
    """Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        convert_to_single_shot (bool, optional):
            Whether to convert the counts and the random basis from multiple shots
            to single shot per snapshot for classical shadow post-processing.
            Default to False.
        rho_method (RhoMKCellMethod, optional):
            The method to use for the calculation. Defaults to "Python_precomputed".

            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The dictionary of rho_m, the sorted list of the selected qubits, and calculation time.
    """

    _measured_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )
    check_random_basis(random_unitary_um, selected_classical_registers)
    if convert_to_single_shot:
        counts, random_unitary_um = spreadout(shots, counts, random_unitary_um)

    begin = time.time()

    selected_clregs_sorted = sorted(selected_classical_registers, reverse=True)
    cell_calculation_method = (
        rho_mk_cell_py_precomputed if rho_method == "numpy_precomputed" else rho_mk_cell_py
    )

    cell_calculation_results = [
        cell_calculation_method(
            idx, single_counts, random_unitary_um[idx], selected_classical_registers
        )
        for idx, single_counts in enumerate(counts)
    ]

    rho_m_list: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = [
        sum(rho_mk * num_bitstring for bitstring, num_bitstring, rho_mk in rho_m_k_data) / shots
        for idx, rho_m_k_data, selected_clregs_sorted_result in cell_calculation_results
    ]  # type: ignore

    problematic_cells = [
        idx
        for idx, rho_m_k_data, selected_clregs_sorted_result in cell_calculation_results
        if selected_clregs_sorted_result != selected_clregs_sorted
    ]
    if problematic_cells:
        warnings.warn(
            f"Selected qubits are not sorted for {problematic_cells} cells.",
            RuntimeWarning,
        )

    taken = time.time() - begin

    return rho_m_list, selected_clregs_sorted, taken


# pylint: disable=invalid-name
RhoMCoreMethod = Union[RhoMKCellMethod, Literal["numpy_flatten", "jax_flatten"], str]
"""Type for rho_m_core method.
It can be either "numpy", "numpy_precomputed" or "numpy_flatten".

- "numpy": Use Numpy to calculate the rho_m.
- "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
- "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.

Currently, "numpy_precomputed" is the best option for performance.
"""
# pylint: enable=invalid-name


def rho_m_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    convert_to_single_shot: bool = False,
    rho_method: RhoMCoreMethod = "numpy_precomputed",
) -> tuple[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]], list[int], float]:
    """Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        convert_to_single_shot (bool, optional):
            Whether to convert the counts and the random basis from multiple shots
            to single shot per snapshot for classical shadow post-processing.
            Default to False.
        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".

            - "numpy":
                Use Numpy to calculate the rho_m.
            - "numpy_precomputed":
                Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten":
                Use Numpy to calculate the rho_m with a flattening workflow.

            Currently, "numpy_precomputed" is the best option for performance.

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The dictionary of rho_m, the sorted list of the selected qubits, and calculation time.
    """
    if rho_method == "numpy_flatten":
        return rho_m_flatten_core(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
            convert_to_single_shot=convert_to_single_shot,
        )

    if rho_method in ["numpy", "numpy_precomputed"]:
        return rho_m_core_py(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
            convert_to_single_shot=convert_to_single_shot,
            rho_method=rho_method,
        )

    raise ValueError(
        f"Unknown rho_method: {rho_method}. "
        "Available methods are: 'numpy', 'numpy_precomputed', 'numpy_flatten'"
    )
