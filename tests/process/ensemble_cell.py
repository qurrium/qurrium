"""Tests qurry.process.utils.randomized module."""

from typing import Union, TypedDict
import os
from itertools import combinations
import json
import pytest
import numpy as np

from qurry.process.utils import randomized_availability, NUMERICAL_ERROR_TOLERANCE
from qurry.process.utils.randomized import ensemble_cell as ensemble_cell_py, ensemble_cell_rust


DUMMY_CASE_FILE = os.path.join(os.path.dirname(__file__), "dummy_data", "ensemble_cell.json")


class EnsembleTarget(TypedDict):
    """The test target type for ensemble_cell function parameters."""

    s_i: str
    """First count's qubits arrange."""
    s_i_meas: int
    """Counting value of the first register."""
    s_j: str
    """Second count's qubits arrange."""
    s_j_meas: int
    """Counting value of the second register."""
    a_num: int
    """Number of qubits."""
    shots: int
    """Number of shots."""


class EnsembleCase(TypedDict):
    """The test case type for ensemble_cell function."""

    target: EnsembleTarget
    """The target parameters for the ensemble_cell function."""
    answer: float
    """The expected answer from the ensemble_cell function."""
    note: str
    """Additional note for the test case."""


with open(DUMMY_CASE_FILE, "r") as f:
    DUMMY_CASES_JSON: list[EnsembleCase] = json.load(f)


ensemble_cases_entries: list[tuple[EnsembleTarget, float]] = [
    (case["target"], case["answer"]) for case in DUMMY_CASES_JSON
]


@pytest.mark.parametrize(["target", "answer"], ensemble_cases_entries)
def test_ensemble_cell_rust(target: EnsembleTarget, answer: float):
    """Test the ensemble_cell_rust function."""

    assert randomized_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {randomized_availability[2]}"
    )

    py_result = ensemble_cell_py(**target)
    rust_result = ensemble_cell_rust(**target)

    comparison_target: list[tuple[str, Union[float, np.float64]]] = [
        ("Python", py_result),
        ("Rust", rust_result),
        ("Answer", answer),
    ]
    for (name_1, result_1), (name_2, result_2) in combinations(comparison_target, 2):
        assert np.abs(result_1 - result_2) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_2} results are not equal in purity_echo_core: "
            f"{name_1}: {result_1}, {name_2}: {result_2}."
        )
