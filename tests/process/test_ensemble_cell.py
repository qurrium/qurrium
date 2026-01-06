"""Tests qurry.process.utils.randomized module."""

from typing import TypedDict
from itertools import combinations
import pytest

from qurry.process.utils import randomized_availability
from qurry.process.utils.randomized import ensemble_cell as ensemble_cell_py, ensemble_cell_rust

from utils import quick_json_read, get_dummy_file_path, numerical_tolerance_check, FloatType


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


DUMMY_CASE_FILE = get_dummy_file_path("ensemble_cell.json")
DUMMY_CASES_JSON: list[EnsembleCase] = quick_json_read(DUMMY_CASE_FILE)
ensemble_cases_entries: list[tuple[EnsembleTarget, float]] = [
    (case["target"], case["answer"]) for case in DUMMY_CASES_JSON
]


def test_availability():
    """Test the availability of the Rust backend for the ensemble_cell function."""

    assert randomized_availability[1]["Rust"] != "Error", (
        f"Rust is not available. Check the error: {randomized_availability[2]}"
    )


@pytest.mark.parametrize(["target", "answer"], ensemble_cases_entries)
def test_ensemble_cell_rust(target: EnsembleTarget, answer: float):
    """Test the ensemble_cell_rust function."""

    py_result = ensemble_cell_py(**target)
    rust_result = ensemble_cell_rust(**target)

    comparison_target: list[tuple[str, FloatType]] = [
        ("Python", py_result),
        ("Rust", rust_result),
        ("Answer", answer),
    ]
    for (name_1, result_1), (name_2, result_2) in combinations(comparison_target, 2):
        assert numerical_tolerance_check(result_1, result_2), (
            f"{name_1} and {name_2} results are not equal in purity_echo_core: "
            f"{name_1}: {result_1}, {name_2}: {result_2}."
        )
