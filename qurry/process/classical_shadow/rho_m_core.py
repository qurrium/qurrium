"""Post Processing - Classical Shadow - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_m_core`)

"""

import time
import warnings
from typing import Literal, Union
import numpy as np

from .rho_mk_cell import rho_mk_cell_py, rho_mk_cell_py_precomputed, RhoMKCellMethod
from ..availability import (
    availablility,
    default_postprocessing_backend,
    PostProcessingBackendLabel,
)
from ..exceptions import (
    # PostProcessingRustImportError,
    PostProcessingRustUnavailableWarning,
)
from ...tools import ParallelManager


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
    ],
)
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(RUST_AVAILABLE, False)


def rho_m_core_py(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    method: RhoMKCellMethod = "Python_precomputed",
    multiprocess: bool = True,
) -> tuple[
    dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    list[int],
    str,
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
        method (RhoMKCellMethod, optional):
            The method to use for the calculation. Defaults to "Python_precomputed".
            It can be either "Python" or "Python_precomputed".
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.

    Returns:
        tuple[
            dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            str,
            float
        ]:
            The rho_m, the set of rho_m_i,
            the sorted list of the selected qubits,
            the message, the taken time.
    """
    sample_shots = sum(counts[0].values())
    assert sample_shots == shots, f"shots {shots} does not match sample_shots {sample_shots}"

    # Determine subsystem size
    measured_system_size = len(list(counts[0].keys())[0])

    if selected_classical_registers is None:
        selected_classical_registers = list(range(measured_system_size))
    elif not isinstance(selected_classical_registers, list):
        raise ValueError(
            "selected_classical_registers should be list, "
            + f"but get {type(selected_classical_registers)}"
        )
    assert all(
        0 <= q_i < measured_system_size for q_i in selected_classical_registers
    ), f"Invalid selected classical registers: {selected_classical_registers}"

    begin = time.time()

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)

    rho_m_dict = {}
    selected_qubits_checked: dict[int, bool] = {}

    cell_calculation_method = (
        rho_mk_cell_py_precomputed if method == "Python_precomputed" else rho_mk_cell_py
    )
    expected_shape = (
        2 ** len(selected_classical_registers_sorted),
        2 ** len(selected_classical_registers_sorted),
    )

    if multiprocess:
        pool = ParallelManager()
        cell_calculation_results = pool.starmap(
            cell_calculation_method,
            [
                (idx, single_counts, random_unitary_um[idx], selected_classical_registers)
                for idx, single_counts in enumerate(counts)
            ],
        )
    else:
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

        tmp_arr = [rho_mk * num_bitstring for bitstring, num_bitstring, rho_mk in rho_m_k_data]
        tmp = sum(tmp_arr) / shots
        assert isinstance(tmp, np.ndarray), f"Invalid rho_m type {type(tmp)} for {idx} cell."

        assert (
            tmp.shape == expected_shape
        ), f"Invalid rho_m shape {tmp.shape}, expected {expected_shape} for {idx} cell."
        rho_m_dict[idx] = tmp

    if any(selected_qubits_checked.values()):
        problematic_cells = [idx for idx, checked in selected_qubits_checked.items() if checked]
        warnings.warn(
            f"Selected qubits are not sorted for {problematic_cells} cells.",
            RuntimeWarning,
        )

    taken = round(time.time() - begin, 3)

    return rho_m_dict, selected_classical_registers_sorted, "", taken


def rho_m_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    backend: Union[PostProcessingBackendLabel, RhoMKCellMethod] = DEFAULT_PROCESS_BACKEND,
    multiprocess: bool = True,
) -> tuple[
    dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    list[int],
    str,
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
        backend (PostProcessingBackendLabel, optional):
            The backend to use for the calculation. Defaults to DEFAULT_PROCESS_BACKEND.
            It can be either "Python" or "Rust".
        multiprocess (bool, optional):
            Whether to use multiprocessing. Defaults to True.

    Returns:
        tuple[
            dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            str,
            float
        ]:
            The rho_m, the set of rho_m_i,
            the sorted list of the selected qubits,
            the message, the taken time.
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

    if backend in ["Python", "Python_precomputed"]:
        return rho_m_core_py(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
            method=backend,  # type: ignore
            multiprocess=multiprocess,
        )

    raise ValueError(f"Invalid backend {backend}. It should be either 'Python' or 'Rust'.")
