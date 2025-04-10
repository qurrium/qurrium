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
