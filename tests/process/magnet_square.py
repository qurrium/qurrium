"""Test qurry.process.magnet_square module."""

from typing import TypedDict, Literal, Union
import os
from itertools import combinations
import json
import pytest
import numpy as np

from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.magnet_square import (
    magnet_square_availability,
    magnet_square,
    z_dir_magnet_square,
    MagnetSquareResult,
)

DUMMY_CASE_FILE_MS = os.path.join(os.path.dirname(__file__), "dummy_data", "magnet_square.json")
DUMMY_CASE_FILE_MSZDIR = os.path.join(
    os.path.dirname(__file__), "dummy_data", "magnet_square_zdir.json"
)


class MagnetSquareZdirTarget(TypedDict):
    """TypedDict for magnet square answer from JSON."""

    shots: int
    """Number of shots."""
    counts: list[dict[str, int]]
    """The counts dictionary list."""
    num_qubits: int
    """Number of qubits."""


class MagnetSquareTarget(MagnetSquareZdirTarget):
    """TypedDict for magnet square answer from JSON."""

    unitary_operator: Union[Literal["x", "y", "z"], str]
    """The unitary operator used."""


class MagnetSquareZdirCase(TypedDict):
    """The raw read magnet square z direction unit type."""

    target: MagnetSquareZdirTarget
    """The target parameters for the z_dir_magnet_square function."""
    answer: MagnetSquareResult
    """The expected answer from the z_dir_magnet_square function."""
    case_name: str
    """The case name."""


class MagnetSquareCase(TypedDict):
    """The raw read magnet square unit type."""

    target: MagnetSquareTarget
    """The target parameters for the magnet_square function."""
    answer: MagnetSquareResult
    """The expected answer from the magnet_square function."""
    case_name: str
    """The case name."""


ANSWERS = {
    "cat_2": 1,
    "cat_4": 1,
    "cat_6": 1,
    "cat_8": 1,
    "trivialPM_2": 1 / 2,
    "trivialPM_4": 1 / 4,
    "trivialPM_6": 1 / 6,
    "trivialPM_8": 1 / 8,
}

ANSWERS_ERROR = 0.05


with open(DUMMY_CASE_FILE_MSZDIR, "r") as f:
    DUMMY_CASES_JSON_MSZDIR: list[MagnetSquareZdirCase] = json.load(f)

mszdir_cases_entries = [
    (case["target"], case["answer"], case["case_name"]) for case in DUMMY_CASES_JSON_MSZDIR
]


with open(DUMMY_CASE_FILE_MS, "r") as f:
    DUMMY_CASES_JSON_MS: list[MagnetSquareCase] = json.load(f)

ms_cases_entries = [
    (case["target"], case["answer"], case["case_name"]) for case in DUMMY_CASES_JSON_MS
]


@pytest.mark.parametrize(["target", "answer", "case_name"], mszdir_cases_entries)
def test_magnet_square_zdir(
    target: MagnetSquareZdirTarget, answer: MagnetSquareResult, case_name: str
):
    """Test the z_dir_magnetic_square_core function."""

    assert magnet_square_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {magnet_square_availability[2]}"
    )

    assert len(target["counts"]) == 1, (
        "The counts should be a single item for the z_dir_magnetic_square_core function."
    )

    py_result = z_dir_magnet_square(
        shots=target["shots"],
        single_counts=target["counts"][0],
        num_qubits=target["num_qubits"],
        backend="Python",
    )
    rust_result = z_dir_magnet_square(
        shots=target["shots"],
        single_counts=target["counts"][0],
        num_qubits=target["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, Union[float, np.float64]]] = [
        ("Python", py_result["magnet_square"]),
        ("Rust", rust_result["magnet_square"]),
        ("Answer", answer["magnet_square"]),
    ]
    for (name_1, result_1), (name_02, result_02) in combinations(comparison_target, 2):
        assert np.abs(result_1 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_02} results are not equal in z_dir_magnetic_square_core: "
            f"{name_1}: {result_1}, {name_02}: {result_02}."
        )
    for name_1, result_1 in comparison_target:
        assert np.abs(result_1 - ANSWERS[case_name]) < ANSWERS_ERROR, (
            f"Result by {name_1} {result_1} is not close to expected "
            f"{ANSWERS[case_name]} in error {ANSWERS_ERROR}."
        )


@pytest.mark.parametrize(["target", "answer", "case_name"], ms_cases_entries)
def test_magnet_square(target: MagnetSquareTarget, answer: MagnetSquareResult, case_name: str):
    """Test the z_dir_magnetic_square_core function."""

    assert magnet_square_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {magnet_square_availability[2]}"
    )

    predict_counts_num = target["num_qubits"] * (target["num_qubits"] - 1)
    assert len(target["counts"]) == predict_counts_num, (
        f"The counts should have {predict_counts_num} items, but got {len(target['counts'])} for {case_name}"
    )

    py_result = magnet_square(
        shots=target["shots"],
        counts=target["counts"],
        num_qubits=target["num_qubits"],
        backend="Python",
    )
    rust_result = magnet_square(
        shots=target["shots"],
        counts=target["counts"],
        num_qubits=target["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, Union[float, np.float64]]] = [
        ("Python", py_result["magnet_square"]),
        ("Rust", rust_result["magnet_square"]),
        ("Answer", answer["magnet_square"]),
    ]
    for (name_1, result_1), (name_02, result_02) in combinations(comparison_target, 2):
        assert np.abs(result_1 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_02} results are not equal in magnet_square_core: "
            f"{name_1}: {result_1}, {name_02}: {result_02}."
        )
    for name_1, result_1 in comparison_target:
        assert np.abs(result_1 - ANSWERS[case_name]) < ANSWERS_ERROR, (
            f"Result by {name_1} {result_1} is not close to expected "
            f"{ANSWERS[case_name]} in error {ANSWERS_ERROR}."
        )
