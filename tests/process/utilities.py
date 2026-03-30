"""Utility functions for tests in the qurry.process module."""

from typing import Any, Literal
import os
import json
import logging
import numpy as np

from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.availability import PostProcessingBackendLabel

FloatType = float | np.float64
"""The type alias for :class:`float` and :class:`~numpy.float64`."""


def get_dummy_file_path(file_name: str) -> str:
    """Get the path to a dummy data file in the tests/process/dummy_data directory.

    Args:
        file_name (str): The name of the dummy data file.

    Returns:
        str: The full path to the dummy data file.
    """

    return os.path.join(os.path.dirname(__file__), "dummy_data", file_name)


def quick_json_read(file_path: str) -> Any:
    """Quickly read a JSON file and return its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Any: The content of the JSON file.
    """

    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def numerical_tolerance_check(
    value1: FloatType, value2: FloatType, tolerance: FloatType = NUMERICAL_ERROR_TOLERANCE
) -> bool:
    """Check if two numerical values are within a specified tolerance.

    Args:
        value1 (FloatType):
            The first numerical value.
        value2 (FloatType):
            The second numerical value.
        tolerance (FloatType):
            The acceptable tolerance level. Defaults to NUMERICAL_ERROR_TOLERANCE.


    Returns:
        bool: True if the values are within the tolerance, False otherwise.
    """

    return np.abs(value1 - value2) <= tolerance


AvailStatusType = tuple[
    str,
    dict[PostProcessingBackendLabel, Literal["Yes", "Error", "Depr.", "No"]],
    dict[PostProcessingBackendLabel, ImportError | None],
]
"""The type alias for availability status list."""


HINT_OF_STATUS = {
    "Yes": "The Rust backend is available.",
    "Error": "There was an error during the Rust backend installation or import.",
    "Depr.": "The Rust backend is deprecated.",
    "No": "The Rust backend is not supported.",
}


def no_error_and_msg_of_availability(
    availability_item: AvailStatusType, backend_label: PostProcessingBackendLabel = "Rust"
) -> tuple[bool, str]:
    """Check if there is no error in the Rust backend availability item.

    Args:
        availability_item (AvailStatusType): The availability item to check.
        backend_label (PostProcessingBackendLabel): The backend label to check. Defaults to "Rust".

    Returns:
        A tuple containing a boolean indicating no error and a message.
    """
    is_no_error = availability_item[1][backend_label] != "Error"
    msg_info = f" - {availability_item[0]} - Status: {availability_item[1][backend_label]}"

    return is_no_error, (
        ("PASS" + msg_info + f" - Hint: {HINT_OF_STATUS[availability_item[1][backend_label]]}")
        if is_no_error
        else ("FAIL" + msg_info + f" - Error message: {availability_item[2][backend_label]}")
    )


def assert_and_logging_rust_available(
    avail_status_list: list[AvailStatusType], logger: logging.Logger | None = None
) -> None:
    """Check if the Rust backend is available.

    Args:
        avail_status_list (list[AvailStatusType]): The availability status list.
    """

    for availability_item in avail_status_list:
        is_no_error, msg = no_error_and_msg_of_availability(availability_item)
        if logger is not None:
            if is_no_error:
                logger.info(msg)
            else:
                logger.error(msg)
        assert is_no_error, msg
