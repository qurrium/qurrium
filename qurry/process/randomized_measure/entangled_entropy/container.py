"""Post Processing - Randomized Measure - Entangled Entropy - Container
(:mod:`qurry.process.randomized_measure.entangled_entropy.container`)

"""

from typing import Union, Optional, Literal, TypedDict
import numpy as np


class TargetSystemResult(TypedDict):
    """The return type of the post-processing for entangled entropy."""

    purity: Union[np.float64, float]
    """The purity of the system."""
    entropy: Union[np.float64, float]
    """The entropy of the system."""
    puritySD: Union[np.float64, float]
    """The standard deviation of the purity."""
    entropySD: Union[np.float64, float]
    """The standard deviation of the entropy."""
    purityCells: Union[dict[int, np.float64], dict[int, float]]
    """The purity of each single count."""

    num_classical_registers: int
    """The number of classical registers."""
    classical_registers: Optional[list[int]]
    """The list of the index of the selected classical registers."""
    classical_registers_actually: list[int]
    """The list of the index of the selected classical registers which is actually used."""

    taking_time: float
    """The calculation time."""
    counts_num: int
    """The number of counts."""


class AllSystemResult(TargetSystemResult):
    """The return type of the post-processing for entangled entropy."""

    preparing_datetime: str
    """The datetime string when preparing the all system result."""
    result_hash_id: str
    """The hash id of the result for verification."""
    all_system_source: Union[str, Literal["independent"]]
    """The name of source of all system.

    - independent: The all system is calculated independently.
    - null_counts: No counts exist.
    """


def isvalid_all_system_result(all_sys_result: AllSystemResult) -> None:
    """Verify if the given AllSystemResult object is valid.

    Args:
        all_sys_result (AllSystemResult):
            The AllSystemResult TypedDict object.
    Raises:
        ValueError: If the all_sys_result argument is not a valid AllSystemResult object.
    """
    if any(
        key not in all_sys_result
        for key in [
            "purity",
            "entropy",
            "puritySD",
            "entropySD",
            "purityCells",
            "num_classical_registers",
            "classical_registers",
            "classical_registers_actually",
            "taking_time",
            "counts_num",
            # all system specific
            "preparing_datetime",
            "result_hash_id",
            "all_system_source",
        ]
    ):
        raise ValueError("The all_sys_result argument must be a valid AllSystemResult object.")
