"""Test qurry.process.hadamard_test module."""

from itertools import combinations
import pytest
import numpy as np

from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.hadamard_test import purity_echo_core_availability
from qurry.process.hadamard_test.purity_echo_core import purity_echo_core


hadamard_cases = [(100, [{"0": 50, "1": 50}], 0), (100, [{"0": 100}], 1), (100, [{"1": 100}], 1)]


@pytest.mark.parametrize(["shots", "counts", "answer"], hadamard_cases)
def test_hadamard(shots: int, counts: list[dict[str, int]], answer: float):
    """Test the purity_echo_core function."""

    assert purity_echo_core_availability[1][
        "Rust"
    ], f"Rust is not available. Check the error: {purity_echo_core_availability[2]}"

    rust_result = purity_echo_core(shots=shots, counts=counts, backend="Rust")
    py_result = purity_echo_core(shots=shots, counts=counts, backend="Python")

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
