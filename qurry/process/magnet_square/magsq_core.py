"""Post Processing - Magnetization Square - Core (:mod:`qurry.process.magnet_square.magsq_core`)"""

import time
import warnings
from typing import Union
from itertools import permutations
from multiprocessing import get_context
import numpy as np

from ..availability import availablility, default_postprocessing_backend, PostProcessingBackendLabel
from ..utils import single_counts_recount
from ..exceptions import PostProcessingRustImportError, PostProcessingRustUnavailableWarning
from ...tools import DEFAULT_POOL_SIZE


try:
    from ...boorust import magnet_square  # type: ignore

    magnetic_square_core_rust_source = magnet_square.magnetic_square_core_rust
    z_dir_magnetic_square_core_rust_source = magnet_square.z_dir_magnetic_square_core_rust

    RUST_AVAILABLE = True
    FAILED_RUST_IMPORT = None
except ImportError as err:
    RUST_AVAILABLE = False
    FAILED_RUST_IMPORT = err

    def magnetic_square_core_rust_source(*args, **kwargs):
        """Dummy function for magnetic_square_core_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate magnetic square."
        ) from FAILED_RUST_IMPORT

    def z_dir_magnetic_square_core_rust_source(*args, **kwargs):
        """Dummy function for z_dir_magnetic_square_core_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate z direction magnetic square."
        ) from FAILED_RUST_IMPORT


BACKEND_AVAILABLE = availablility(
    "magnet_square.magnsq_core", [("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT)]
)
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(RUST_AVAILABLE, False)


def magsq_cell_py_deprecated(
    idx: int, single_counts: dict[str, int], shots: int
) -> tuple[int, np.float64]:
    """Calculate the magnitudes square cell

    Args:
        idx (int): Index of the cell (counts).
        single_counts (dict[str, int]): Single counts of the cell.
        shots (int): Shots of the experiment on quantum machine.

    Returns:
        tuple[int, np.float64]: Index, one of magnitudes square.
    """

    magnetsq_cell = np.float64(0)
    for bits in single_counts:
        ratio = np.float64(single_counts[bits]) / shots
        magnetsq_cell += ratio if bits[0] == bits[1] else -ratio
    return idx, magnetsq_cell


def magsq_cell_py(idx: int, single_counts: dict[str, int], shots: int) -> tuple[int, np.float64]:
    """Calculate the magnitudes square cell

    Args:
        idx (int): Index of the cell (counts).
        single_counts (dict[str, int]): Single counts of the cell.
        shots (int): Shots of the experiment on quantum machine.

    Returns:
        tuple[int,  np.float64]: Index, one of magnitudes square.
    """

    magnetsq_cell = sum(
        np.float64(c) * (1 if bits[0] == bits[1] else -1) / shots
        for bits, c in single_counts.items()
    ) + np.float64(0)

    return idx, magnetsq_cell


def magsq_cell_wrapper(arguments: tuple[int, dict[str, int], int]) -> tuple[int, np.float64]:
    """Wrapper for the magnetic square cell.

    Args:
        arguments (tuple[int, dict[str, int], int, PostProcessingBackendLabel]):
            The arguments for the magnetic square cell.
            - idx (int): Index of the cell (counts).
            - single_counts (dict[str, int]): Single counts of the cell.
            - shots (int): Shots of the experiment on quantum machine.

    Returns:
        tuple[int, np.float64]: Index, one of magnitudes square.
    """
    return magsq_cell_py(*arguments)


def magnetic_square_core(
    shots: int,
    counts: list[dict[str, int]],
    num_qubits: int,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> tuple[Union[float, np.float64], Union[dict[int, float], dict[int, np.float64]], float]:
    """The core function of Magnetization square by Python.

    Args:
        shots (int): Shots of the experiment on quantum machine.
        counts (list[dict[str, int]]): Counts of the experiment on quantum machine.
        num_qubits (int): Number of qubits.
        backend (PostProcessingBackendLabel, optional):
            Post Processing backend. Defaults to DEFAULT_PROCESS_BACKEND.

    Returns:
        tuple[Union[float, np.float64], Union[dict[int, float], dict[int, np.float64]], float]:
            Magnetization square, magnetization square cell, time taken.
    """

    if len(counts) != num_qubits * (num_qubits - 1):
        raise ValueError(
            f"Counts length {len(counts)} must be equal to "
            f"num_qubits * (num_qubits - 1) = {num_qubits * (num_qubits - 1)}."
        )

    if backend == "Rust":
        if RUST_AVAILABLE:
            return magnetic_square_core_rust_source(shots, counts, num_qubits)
        warnings.warn(
            PostProcessingRustUnavailableWarning(
                "Rust is not available, using python to calculate magnetic square."
            )
        )

    sample_counts_sum = sum(counts[0].values())
    assert (
        shots == sample_counts_sum
    ), f"Shots: {shots} must be equal to the sum of counts: {sample_counts_sum}."
    assert all(len(bits) == 2 for bits in counts[0]), f"Bits must be 2 bit, but found: {counts[0]}"

    begin = time.time()
    pool = get_context("spawn").Pool(DEFAULT_POOL_SIZE)
    with pool as p:
        magnetsq_cell_dict = dict(
            p.map(magsq_cell_wrapper, [(i, c, shots) for i, c in enumerate(counts)])
        )
    magnetsq = np.float64(sum(magnetsq_cell_dict.values()) + num_qubits) / (num_qubits**2)
    taken = round(time.time() - begin, 3)

    return magnetsq, magnetsq_cell_dict, taken


def z_dir_magnetic_square_core(
    shots: int,
    single_counts: dict[str, int],
    num_qubits: int,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> tuple[Union[float, np.float64], Union[dict[int, float], dict[int, np.float64]], float]:
    """The core function of Z direction Magnetization square by Python.

    Args:
        shots (int): Shots of the experiment on quantum machine.
        single_counts (dict[str, int]): Single count.
        num_qubits (int): Number of qubits.
        backend (PostProcessingBackendLabel, optional):
            Post Processing backend. Defaults to DEFAULT_PROCESS_BACKEND.

    Returns:
        tuple[Union[float, np.float64], Union[dict[int, float], dict[int, np.float64]], float]:
            Magnetization square, magnetization square cell, time taken.
    """

    if backend == "Rust":
        if RUST_AVAILABLE:
            return z_dir_magnetic_square_core_rust_source(shots, single_counts, num_qubits)
        warnings.warn(
            PostProcessingRustUnavailableWarning(
                "Rust is not available, using python to calculate magnetic square."
            )
        )

    sample_counts_sum = sum(single_counts.values())
    assert (
        shots == sample_counts_sum
    ), f"Shots: {shots} must be equal to the sum of counts: {sample_counts_sum}."

    begin = time.time()
    pool = get_context("spawn").Pool(DEFAULT_POOL_SIZE)
    with pool as p:
        magnetsq_cell_dict = dict(
            p.map(
                magsq_cell_wrapper,
                [
                    (idx, single_counts_recount(single_counts, num_qubits, [i, j]), shots)
                    for idx, (i, j) in enumerate(permutations(range(num_qubits), 2))
                ],
            )
        )
    magnetsq = np.float64(sum(magnetsq_cell_dict.values()) + num_qubits) / (num_qubits**2)
    taken = round(time.time() - begin, 3)

    return magnetsq, magnetsq_cell_dict, taken
