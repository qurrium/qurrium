"""Post Processing - Magnetization Square - Cell (:mod:`qurry.process.magnet_square.magsq_cell`)"""

import numpy as np


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
