"""Utility functions for testing qurry package."""

from typing import TypedDict, Any, Optional
import numpy as np

from qurry.tools.datetime import current_time


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
        + f"or other quantities: {other_quantity_names}"
        if other_quantity_names
        else "" + " not found in quantity." + f" Quantity: {quantity}"
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
