"""Tests qurry.process.utils.construct module."""

import os
import pytest

from qurry.capsule import quickRead
from qurry.process.utils import construct_availability
from qurry.process.utils.construct import (
    counts_under_degree as counts_under_degree_py,
    counts_under_degree_rust,
)


FILE_LOCATION = os.path.join(os.path.dirname(__file__), "easy-dummy.json")

easy_dummy: dict[str, dict[str, int]] = quickRead(FILE_LOCATION)

test_setup_counts_substring: list[list[int]] = (
    [[i] for i in range(8)]
    + [[i, i + 1] for i in range(7)]
    + [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
)


@pytest.mark.parametrize("test_items", test_setup_counts_substring)
def test_counts_substring(test_items: list[int]):
    """Test the ensemble_cell_rust function."""

    assert construct_availability[1]["Rust"], (
        "Rust is not available." + f" Check the error: {construct_availability[2]}"
    )

    counts_under_degree_py_result = counts_under_degree_py(easy_dummy["0"], 8, test_items)
    counts_under_degree_rust_result = counts_under_degree_rust(easy_dummy["0"], 8, test_items)

    assert all(
        counts_under_degree_rust_result[s] == v for s, v in counts_under_degree_py_result.items()
    ), (
        "Rust and Python results are not equal in counts_under_degree. "
        + f"test_items: {test_items}, "
        + f"counts_under_degree_rust_result: {counts_under_degree_rust_result}, "
        + f"counts_under_degree_py_result: {counts_under_degree_py_result}."
    )
