"""Post Processing - String Operator - String Operator
(:mod:`qurry.process.string_operator.string_operator`)

"""

from typing import Union, Optional, TypedDict
import numpy as np
import tqdm

from ..availability import PostProcessingBackendLabel
from .strop_core import string_operator_core, DEFAULT_PROCESS_BACKEND


class StringOperator(TypedDict):
    """String Operator type."""

    order: Union[float, np.float64]
    """The order of the string operator."""


def string_operator_order(
    shots: int,
    counts: list[dict[str, int]],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
    pbar: Optional[tqdm.tqdm] = None,
) -> StringOperator:
    """Calculate the order of the string operator.

    Reference:
        .. note::
            - Crossing a topological phase transition with a quantum computer -
            Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank,
            [PhysRevResearch.4.L022020](https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020)

        .. code-block:: bibtex
            @article{PhysRevResearch.4.L022020,
                title = {Crossing a topological phase transition with a quantum computer},
                author = {Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank},
                journal = {Phys. Rev. Research},
                volume = {4},
                issue = {2},
                pages = {L022020},
                numpages = {8},
                year = {2022},
                month = {Apr},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevResearch.4.L022020},
                url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020}
            }

    Args:
        shots (int): Number of shots.
        counts (list[dict[str, int]]): List of counts.
        backend (Optional[PostProcessingBackendLabel], optional): Backend to use. Defaults to None.
        pbar (Optional[tqdm.tqdm], optional): Progress bar. Defaults to None.

    Returns:
        StringOperator: String Operator.
    """
    if isinstance(pbar, tqdm.tqdm):
        pbar.set_description("String Operator being calculated.")
    order = string_operator_core(
        shots=shots,
        counts=counts,
        backend=backend,
    )
    return {
        "order": order,
    }
