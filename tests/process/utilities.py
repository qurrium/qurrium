"""Utility functions for tests in the qurry.process module."""

from typing import Any, Literal
import os
import json
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


def assert_rust_available(avail_status_list: list[AvailStatusType]):
    """Check if the Rust backend is available.

    Args:
        avail_status_list (list[AvailStatusType]): The availability status list.
    """

    for availability_item in avail_status_list:
        assert availability_item[1]["Rust"] != "Error", (
            f"Rust is not available. Check the error: {availability_item[2]}"
        )
