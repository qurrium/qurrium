"""Utility functions for testing qurry package."""

import os
from typing import TypedDict, Any, Optional
import warnings
import numpy as np

from qurry.capsule import quickRead
from qurry.tools.datetime import current_time
from qurry.tools.backend.import_simulator import SIM_DEFAULT_SOURCE, SIM_IMPORT_ERROR_INFOS
from qurry.exceptions import QurryDependenciesNotWorking

SEED_FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")


def detect_simulator_source() -> str:
    """Detect the simulator source.
    If the default simulator source is not Qiskit Aer, a warning is raised.
    This function is used to check if the Qiskit Aer simulator is available.

    Returns:
        str: The simulator source.
    """

    if SIM_DEFAULT_SOURCE != "qiskit_aer":
        warnings.warn(
            f"Qiskit Aer is not used as the default simulator: {SIM_DEFAULT_SOURCE}. "
            f"Current simulator source is: {SIM_IMPORT_ERROR_INFOS[SIM_DEFAULT_SOURCE]},"
            "some test cases may be skipped.",
            category=QurryDependenciesNotWorking,
        )
    return SIM_DEFAULT_SOURCE


def prepare_random_unitary_seeds(
    filename: str = SEED_FILE_LOCATION,
) -> dict[int, dict[int, dict[int, int]]]:
    """Prepare random unitary seeds from a file.

    Args:
        filename (str): The filename containing the random unitary seeds.

    Returns:
        dict[str, dict[str, dict[str, int]]]: The random unitary seeds.
    """

    random_unitary_seeds_raw: dict[str, dict[str, dict[str, int]]] = quickRead(filename)
    random_unitary_seeds = {
        int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
        for k, v in random_unitary_seeds_raw.items()
    }
    return random_unitary_seeds


def current_time_filename():
    """Returns the current time as a filename.

    Returns:
        str: The current time as a filename.
    """
    return current_time().replace(":", "").replace("-", "").replace(" ", "_")


class InputUnit(TypedDict):
    """Test unit."""

    measure: dict[str, Any]
    analyze: dict[str, Any]
    answer: float


class ResultUnit(TypedDict):
    """Result unit."""

    answer: float
    diff: float
    target_quantity: float
    target_quantity_name: str
    is_correct: bool


def check_unit(
    quantity: dict[str, Any],
    target_quantity_name: str,
    answer: float,
    threshold: float,
    test_item_name: str,
    other_quantity_names: Optional[list[str]] = None,
) -> ResultUnit:
    """Check the unit of the test.

    Args:
        quantity (dict[str, Any]):
            The quantity to check.
        target_quantity_name (str):
            The name of the target quantity.
        answer (float):
            The expected answer.
        threshold (float):
            The threshold for the check.
        test_item_name (str):
            The name of the test item.
        other_quantity_names (Optional[list[str]]):
            Other quantities to check.

    Returns:
        ResultUnit: The result of the check.
    """

    assert all(
        [target_quantity_name in quantity]
        + ([k in quantity for k in other_quantity_names] if other_quantity_names else [])
    ), (
        f"{test_item_name} | The necessary quantities '{target_quantity_name}' "
        + (f"or other quantities: {other_quantity_names}" if other_quantity_names else "")
        + f" not found in quantity. Quantity: {quantity}"
    )

    diff = np.abs(quantity[target_quantity_name] - answer)
    is_correct = diff < threshold
    assert is_correct, (
        f"{test_item_name} | The result of '{target_quantity_name}' is not correct: "
        + f"{diff} !< {threshold}, {quantity[target_quantity_name]} != {answer}."
    )

    return ResultUnit(
        answer=quantity[target_quantity_name],
        diff=diff,
        target_quantity=answer,
        target_quantity_name=target_quantity_name,
        is_correct=is_correct,
    )
