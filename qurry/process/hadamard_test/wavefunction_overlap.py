"""Post Processing - Hadamard Test - Wavefunction Overlap
(:mod:`qurry.process.hadamard_test.wavefunction_overlap`)

"""

from typing import Union, Optional, TypedDict
import tqdm
import numpy as np


from ..availability import PostProcessingBackendLabel
from .purity_echo_core import purity_echo_core, DEFAULT_PROCESS_BACKEND


class HadamardOverlapResult(TypedDict):
    """The return type of the post-processing for wavefunction overlap."""

    echo: Union[np.float64, float]
    """The Loschmidt Echo."""


def hadamard_overlap_echo(
    shots: int,
    counts: list[dict[str, int]],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    pbar: Optional[tqdm.tqdm] = None,
) -> HadamardOverlapResult:
    """Calculate overlap echo with more information combined.
    The echo we compute is the Loschmidt Echo.

    Args:
        shots (int):
            Shots of the experiment on quantum machine.
        counts (list[dict[str, int]]):
            Counts of the experiment on quantum machine.
        backend (PostProcessingBackendLabel, optional):
            Backend of the postprocessing. Defaults to DEFAULT_PROCESS_BACKEND.
        pbar (Optional[tqdm.tqdm], optional):
            Progress bar. Defaults to None.

    Raises:
        ValueError: Get degree neither 'int' nor 'tuple[int, int]'.
        ValueError: Measure range does not contain subsystem.

    Returns:
        Quantity of the experiment.
    """

    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description_str("Calculate entropy by Hadamard Test.")

    return {
        "echo": purity_echo_core(shots, counts, backend),
    }
