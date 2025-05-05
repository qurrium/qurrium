"""Post Processing - Classical Shadow - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_m_core`)

"""

import time
import warnings
from typing import Literal, Union
import numpy as np

from .matrix_calcution import JAX_AVAILABLE, FAILED_JAX_IMPORT
from .rho_mk_cell import rho_mk_cell_py, rho_mk_cell_py_precomputed, RhoMKCellMethod
from .rho_m_flatten import rho_m_flatten_core
from ..utils import shot_counts_selected_clreg_checker_pyrust
from ..availability import (
    availablility,
    PostProcessingBackendLabel,
    default_postprocessing_backend,
)
from ..exceptions import (
    # PostProcessingRustImportError,
    PostProcessingRustUnavailableWarning,
)


# try:

#     from ...boorust import shadow  # type: ignore

#     rho_m_core_rust_source = shadow.rho_m_core_rust

#     RUST_AVAILABLE = True
#     FAILED_RUST_IMPORT = None
# except ImportError as err:
#     RUST_AVAILABLE = False
#     FAILED_RUST_IMPORT = err

#     def rho_m_core_rust_source(*args, **kwargs):
#         """Dummy function for purity_cell_rust."""
#         raise PostProcessingRustImportError(
#             "Rust is not available, using python to calculate purity cell."
#         ) from FAILED_RUST_IMPORT

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
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(RUST_AVAILABLE, False)


def rho_m_core_py(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
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
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.
        rho_method (RhoMKCellMethod, optional):
            The method to use for the calculation. Defaults to "Python_precomputed".
            It can be either "numpy" or "numpy_precomputed".

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

    begin = time.time()

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)

    rho_m_list = []
    selected_qubits_checked: dict[int, bool] = {}

    cell_calculation_method = (
        rho_mk_cell_py_precomputed if rho_method == "numpy_precomputed" else rho_mk_cell_py
    )

    cell_calculation_results = [
        cell_calculation_method(
            idx, single_counts, random_unitary_um[idx], selected_classical_registers
        )
        for idx, single_counts in enumerate(counts)
    ]

    for idx, rho_m_k_data, selected_classical_registers_sorted_result in cell_calculation_results:
        selected_qubits_checked[idx] = (
            selected_classical_registers_sorted_result != selected_classical_registers_sorted
        )

        tmp_arr: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = [
            rho_mk * num_bitstring for bitstring, num_bitstring, rho_mk in rho_m_k_data
        ]
        tmp = sum(tmp_arr) / shots
        rho_m_list.append(tmp)

    if any(selected_qubits_checked.values()):
        problematic_cells = [idx for idx, checked in selected_qubits_checked.items() if checked]
        warnings.warn(
            f"Selected qubits are not sorted for {problematic_cells} cells.",
            RuntimeWarning,
        )

    taken = time.time() - begin

    return rho_m_list, selected_classical_registers_sorted, taken


# pylint: disable=invalid-name
RhoMCoreMethod = Union[RhoMKCellMethod, Literal["numpy_flatten", "jax_flatten"]]
"""Type for rho_m_core method.
It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
- "numpy": Use Numpy to calculate the rho_m.
- "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
- "jax_flatten": Use JAX to calculate the rho_m with a flattening workflow.
- "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
Currently, "numpy_precomputed" is the best option for performance.
"""
# pylint: enable=invalid-name


def rho_m_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
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
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.
        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "jax_flatten": Use JAX to calculate the rho_m with a flattening workflow.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The dictionary of rho_m, the sorted list of the selected qubits, and calculation time.
    """
    if backend == "Rust":
        # if RUST_AVAILABLE:
        #     return rho_m_core_rust_source(
        #         shots=shots,
        #         counts=counts,
        #         random_unitary_um=random_unitary_um,
        #         selected_classical_registers=selected_classical_registers,
        #     )
        warnings.warn(
            "Rust is not available, using Python to calculate purity cell."
            + f"Check the error: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
        backend = "Python"
    if rho_method == "jax_flatten":
        if JAX_AVAILABLE:
            return rho_m_flatten_core(
                shots=shots,
                counts=counts,
                random_unitary_um=random_unitary_um,
                selected_classical_registers=selected_classical_registers,
                method="jax",
            )
        warnings.warn(
            "JAX is not available, using Python to calculate rho_m_flatten.",
            PostProcessingRustUnavailableWarning,
        )
        rho_method = "numpy_flatten"
    if rho_method == "numpy_flatten":
        return rho_m_flatten_core(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
            method="numpy",
        )

    if rho_method in ["numpy", "numpy_precomputed"]:
        return rho_m_core_py(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
            rho_method=rho_method,
        )

    raise ValueError(f"Invalid backend {backend}. It should be either 'Python' or 'Rust'.")
