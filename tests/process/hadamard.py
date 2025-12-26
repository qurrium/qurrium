"""Test qurry.process.hadamard_test module."""

from typing import TypedDict
import os
from itertools import combinations
import json
import pytest
import numpy as np

from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.hadamard_test import purity_echo_core_availability
from qurry.process.hadamard_test.purity_echo_core import purity_echo_core


DUMMY_CASE_FILE = os.path.join(os.path.dirname(__file__), "dummy_data", "hadamard.json")


class HadamardTarget(TypedDict):
    """The test target type for hadamard_test purity_echo_core function parameters."""

    shots: int
    """Number of shots."""
    counts: list[dict[str, int]]
    """The counts dictionary list."""


class HadamardCase(TypedDict):
    """The test case type for hadamard_test purity_echo_core function."""

    target: HadamardTarget
    """The target parameters for the purity_echo_core function."""
    answer: float
    """The expected answer from the purity_echo_core function."""
    note: str
    """Additional note for the test case."""


with open(DUMMY_CASE_FILE, "r") as f:
    DUMMY_CASES_JSON: list[HadamardCase] = json.load(f)

hadamard_cases_entries = [(case["target"], case["answer"]) for case in DUMMY_CASES_JSON]


@pytest.mark.parametrize(["target", "answer"], hadamard_cases_entries)
def test_hadamard(target: HadamardTarget, answer: float):
    """Test the purity_echo_core function."""

    assert purity_echo_core_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {purity_echo_core_availability[2]}"
    )

    rust_result = purity_echo_core(**target, backend="Rust")
    py_result = purity_echo_core(**target, backend="Python")

    comparison_target: list[tuple[str, float]] = [
        ("Python", py_result),
        ("Rust", rust_result),
        ("Answer", answer),
    ]
    for (name_1, result_1), (name_2, result_2) in combinations(comparison_target, 2):
        assert np.abs(result_1 - result_2) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_2} results are not equal in purity_echo_core: "
            f"{name_1}: {result_1}, {name_2}: {result_2}."
        )
