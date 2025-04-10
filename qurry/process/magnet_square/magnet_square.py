"""Post Processing - Magnetic Square - Magnetic Square
(:mod:`qurry.process.magnet_square.magnet_square`)

"""

from typing import Union, Optional, TypedDict
import numpy as np
import tqdm

from ..availability import PostProcessingBackendLabel
from .magsq_core import magnetic_square_core, DEFAULT_PROCESS_BACKEND


class MagnetSquare(TypedDict):
    """Magnetic Square type."""

    magnet_square: Union[float, np.float64]
    """Magnetic Square."""
    magnet_square_cells: dict[int, Union[float, np.float64]]
    """Magnetic Square cells."""
    counts_num: int
    """Number of counts."""
    taking_time: float
    """Taking time."""


def magnet_square(
    shots: int,
    counts: list[dict[str, int]],
    num_qubits: int,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    pbar: Optional[tqdm.tqdm] = None,
) -> MagnetSquare:
    """Calculate the magnet square.

    Args:
        shots (int): Number of shots.
        counts (list[dict[str, int]]): List of counts.
        num_qubits (int): Number of qubits.
        backend (Optional[PostProcessingBackendLabel], optional): Backend to use. Defaults to None.
        pbar (Optional[tqdm.tqdm], optional): Progress bar. Defaults to None.

    Returns:
        MagnetSquare: Magnetic Square.
    """
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description("Magnetic Square being calculated.")
    (magsq, magnet_square_cells, counts_num, taking_time, _msg) = magnetic_square_core(
        shots=shots,
        counts=counts,
        num_qubits=num_qubits,
        backend=backend,
    )
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description(f"Magnetic Square calculated in {taking_time} seconds.")
    return {
        "magnet_square": magsq,
        "magnet_square_cells": magnet_square_cells,
        "counts_num": counts_num,
        "taking_time": taking_time,
    }
