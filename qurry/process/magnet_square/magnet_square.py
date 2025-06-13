"""Post Processing - Magnetization Square - Magnetization Square
(:mod:`qurry.process.magnet_square.magnet_square`)

"""

from typing import Union, Optional, TypedDict
import numpy as np
import tqdm

from ..availability import PostProcessingBackendLabel
from .magsq_core import magnetic_square_core, z_dir_magnetic_square_core, DEFAULT_PROCESS_BACKEND


class MagnetSquare(TypedDict):
    """Magnetization Square type."""

    magnet_square: Union[float, np.float64]
    """Magnetization Square."""
    magnet_square_cells: Union[dict[int, float], dict[int, np.float64]]
    """Magnetization Square cells."""
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
        MagnetSquare: Magnetization Square.
    """
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description("Magnetization Square being calculated.")

    magsq, magnet_square_cells, taking_time = magnetic_square_core(
        shots=shots, counts=counts, num_qubits=num_qubits, backend=backend
    )
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description(f"Magnetization Square calculated in {taking_time} seconds.")

    return {
        "magnet_square": magsq,
        "magnet_square_cells": magnet_square_cells,
        "taking_time": taking_time,
    }


def z_dir_magnet_square(
    shots: int,
    single_counts: dict[str, int],
    num_qubits: int,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    pbar: Optional[tqdm.tqdm] = None,
) -> MagnetSquare:
    """Calculate the magnet square for Z direction.

    Signle counts is only working for Z direction.

    Args:
        shots (int): Number of shots.
        single_counts (dict[str, int]): Single count.
        num_qubits (int): Number of qubits.
        backend (Optional[PostProcessingBackendLabel], optional): Backend to use. Defaults to None.
        pbar (Optional[tqdm.tqdm], optional): Progress bar. Defaults to None.

    Returns:
        MagnetSquare: Magnetization Square.
    """
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description("Z Direction Magnetization Square being calculated.")

    magsq, magnet_square_cells, taking_time = z_dir_magnetic_square_core(
        shots=shots, single_counts=single_counts, num_qubits=num_qubits, backend=backend
    )

    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description(
            f"Z Direction Magnetization Square calculated in {taking_time} seconds."
        )

    return {
        "magnet_square": magsq,
        "magnet_square_cells": magnet_square_cells,
        "taking_time": taking_time,
    }
