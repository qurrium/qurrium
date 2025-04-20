"""Post Processing - Classical Shadow - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_m_core`)

"""

import time
import warnings
from typing import Literal, Union
import numpy as np

from .rho_mk_cell import rho_mk_cell_py
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
    msg = f"| Selected classical registers: {selected_classical_registers}"

    begin = time.time()

    pool = ParallelManager()
    rho_mk_py_result_list = pool.starmap(
        rho_mk_cell_py,
        [
            (idx, single_counts, random_unitary_um[idx], selected_classical_registers)
            for idx, single_counts in enumerate(counts)
        ],
    )

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)

    rho_m_dict = {
        idx: np.zeros(
            (
                2 ** len(selected_classical_registers_sorted),
                2 ** len(selected_classical_registers_sorted),
            ),
            dtype=np.complex128,
        )
        for idx in range(len(rho_mk_py_result_list))
    }
    selected_qubits_checked: dict[int, bool] = {}
    for (
        idx,
        rho_mk_dict,
        rho_mk_counts_num,
        selected_classical_registers_sorted_result,
    ) in rho_mk_py_result_list:

        for bitstring, rho_mk in rho_mk_dict.items():
            rho_m_dict[idx] += rho_mk * rho_mk_counts_num[bitstring]
        rho_m_dict[idx] /= shots
        selected_qubits_checked[idx] = (
            selected_classical_registers_sorted_result != selected_classical_registers_sorted
        )

    if any(selected_qubits_checked.values()):
        problematic_cells = [idx for idx, checked in selected_qubits_checked.items() if checked]
        warnings.warn(
            f"Selected qubits are not sorted for {problematic_cells} cells.",
            RuntimeWarning,
        )

    taken = round(time.time() - begin, 3)

    return rho_m_dict, selected_classical_registers_sorted, msg, taken


def rho_m_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: list[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
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
    if backend == "Python":
        return rho_m_core_py(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_um,
            selected_classical_registers=selected_classical_registers,
        )
    raise ValueError(f"Invalid backend {backend}. It should be either 'Python' or 'Rust'.")
