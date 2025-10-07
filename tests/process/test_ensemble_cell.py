"""Tests qurry.process.utils.randomized module."""

from itertools import combinations
import pytest
import numpy as np

from qurry.process.utils import randomized_availability, NUMERICAL_ERROR_TOLERANCE
from qurry.process.utils.randomized import ensemble_cell as ensemble_cell_py, ensemble_cell_rust


EnsembleTarget = tuple[str, int, str, int, int, int]
"""Target type for ensemble_cell function parameters.

The tuple contains:
- str: Bitstring of the first register.
- int: Integer value of the first register.
- str: Bitstring of the second register.
- int: Integer value of the second register.
- int: Number of qubits.
- int: Number of shots.
"""

ensemble_cases: list[tuple[EnsembleTarget, float]] = [
    (
        ("10010101", 421, "10010101", 421, 8, 4096),
        (np.float64(421) ** 2) / np.float_power(2, 16, dtype=np.float64),
        # (2**8)((-2)**(-0))(421/(2**12))(421/(2**12))
    ),
    (
        ("10010101", 421, "00000000", 11, 8, 4096),
        (np.float64(421) * np.float64(11)) / np.float_power(2, 20, dtype=np.float64),
        # (2**8)((-2)**(-4))(421/(2**12))(11/(2**12))
    ),
]


@pytest.mark.parametrize(["target", "answer"], ensemble_cases)
def test_ensemble_cell_rust(target: EnsembleTarget, answer: float):
    """Test the ensemble_cell_rust function."""

    assert randomized_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {randomized_availability[2]}"
    )

    py_result = ensemble_cell_py(*target)
    rust_result = ensemble_cell_rust(*target)

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
