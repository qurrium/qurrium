"""Utility functions for testing qurry package."""

from typing import Any, Union
import os
import json
import numpy as np

from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE

FloatType = Union[float, np.float64]
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
