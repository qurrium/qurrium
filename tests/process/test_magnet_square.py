"""Test qurry.process.magnet_square module."""

from typing import TypedDict, TypeVar, Literal, Union, overload
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.magnet_square import magnet_square_availability
from qurry.process.magnet_square.magsq_core import magnetic_square_core, z_dir_magnetic_square_core

FILE_LOCATION_MSZDIR = os.path.join(os.path.dirname(__file__), "mszdir-case.json")
FILE_LOCATION_MS = os.path.join(os.path.dirname(__file__), "ms-case.json")


class RawReadMSAnswer(TypedDict):
    """TypedDict for magnet square answer from JSON."""

    magnet_square: float
    num_qubits: int
    shots: int
    magnet_square_cell: dict[str, float]
    unitary_operator: Union[Literal["x", "y", "z"], str]
    taking_time: float


class RawReadMSZDirAnswer(TypedDict):
    """TypedDict for magnet square answer from JSON."""

    magnet_square: float
    num_qubits: int
    shots: int
    magnet_square_cell: dict[str, float]
    taking_time: float


RRMSA = TypeVar("RRMSA", RawReadMSAnswer, RawReadMSZDirAnswer)


# This won;t work with TypedDicts berfore Python3.11,
# so we use a regular TypedDict.
# class RawReadMSUnit(TypedDict, Generic[RRMSA]):
#     """TypedDict for magnet square unit from JSON."""

#     ans: RRMSA
#     counts: list[dict[str, int]]


class RawReadMS(TypedDict):
    """TypedDict for magnet square unit from JSON."""

    ans: RawReadMSAnswer
    counts: list[dict[str, int]]


class RawReadMSZDir(TypedDict):
    """TypedDict for magnet square unit from JSON."""

    ans: RawReadMSZDirAnswer
    counts: list[dict[str, int]]


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


@overload
def raw_unfold_and_sorted(
    case_set: dict[str, RawReadMSZDir],
) -> list[tuple[str, RawReadMSZDirAnswer, list[dict[str, int]]]]: ...


@overload
def raw_unfold_and_sorted(
    case_set: dict[str, RawReadMS],
) -> list[tuple[str, RawReadMSAnswer, list[dict[str, int]]]]: ...


def raw_unfold_and_sorted(case_set):
    """Unfold and sort the raw read magnet square case set.
    Args:
        case_set (dict[str, RawReadMSUnit]): The raw read magnet square case set.
    Returns:
        list[tuple[str, RRMSA, list[dict[str, int]]]]:
            The sorted list of tuples containing case name, answer, and counts.
    """
    return sorted(((k, v["ans"], v["counts"]) for k, v in case_set.items()), key=lambda x: x[0])


raw_mszdir_case: dict[str, RawReadMS] = quickRead(FILE_LOCATION_MSZDIR)
raw_ms_case: dict[str, RawReadMSZDir] = quickRead(FILE_LOCATION_MS)


@pytest.mark.parametrize(["case_name", "answer", "counts"], raw_unfold_and_sorted(raw_mszdir_case))
def test_magnet_square_zdir(
    case_name: str, answer: RawReadMSZDirAnswer, counts: list[dict[str, int]]
):
    """Test the z_dir_magnetic_square_core function."""

    assert magnet_square_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {magnet_square_availability[2]}"
    )

    assert len(counts) == 1, (
        "The counts should be a single item for the z_dir_magnetic_square_core function."
    )

    py_result = z_dir_magnetic_square_core(
        shots=answer["shots"],
        single_counts=counts[0],
        num_qubits=answer["num_qubits"],
        backend="Python",
    )
    rust_result = z_dir_magnetic_square_core(
        shots=answer["shots"],
        single_counts=counts[0],
        num_qubits=answer["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, float]] = [
        ("Python", py_result[0]),
        ("Rust", rust_result[0]),
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


@pytest.mark.parametrize(["case_name", "answer", "counts"], raw_unfold_and_sorted(raw_ms_case))
def test_magnet_square(case_name: str, answer: RawReadMSAnswer, counts: list[dict[str, int]]):
    """Test the z_dir_magnetic_square_core function."""

    assert magnet_square_availability[1]["Rust"], (
        f"Rust is not available. Check the error: {magnet_square_availability[2]}"
    )

    predict_counts_num = answer["num_qubits"] * (answer["num_qubits"] - 1)
    assert len(counts) == predict_counts_num, (
        f"The counts should have {predict_counts_num} items, but got {len(counts)} for {case_name}"
    )

    py_result = magnetic_square_core(
        shots=answer["shots"],
        counts=counts,
        num_qubits=answer["num_qubits"],
        backend="Python",
    )
    rust_result = magnetic_square_core(
        shots=answer["shots"],
        counts=counts,
        num_qubits=answer["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, float]] = [
        ("Python", py_result[0]),
        ("Rust", rust_result[0]),
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
