"""Test qurry.process.magnet_square module."""

from typing import TypedDict, TypeVar, Generic, Literal, Union
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


class RawReadMSZdirAnswer(TypedDict):
    """TypedDict for magnet square answer from JSON."""

    magnet_square: float
    num_qubits: int
    shots: int
    magnet_square_cell: dict[str, float]
    taking_time: float


RRMSA = TypeVar("RRMSA", RawReadMSAnswer, RawReadMSZdirAnswer)


class RawReadMSUnit(TypedDict, Generic[RRMSA]):
    """TypedDict for magnet square unit from JSON."""

    ans: RRMSA
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

raw_mszdir_case: dict[str, RawReadMSUnit[RawReadMSZdirAnswer]] = quickRead(FILE_LOCATION_MSZDIR)
raw_ms_case: dict[str, RawReadMSUnit[RawReadMSAnswer]] = quickRead(FILE_LOCATION_MS)


def test_availability():
    """Test the availability of the Rust backend for the entangled_entropy_core function."""

    for availability_item in [magnet_square_availability]:
        assert availability_item[1]["Rust"], (
            "Rust is not available." + f" Check the error: {availability_item[2]}"
        )


@pytest.mark.parametrize("mszdir_case_item", sorted(raw_mszdir_case.items(), key=lambda x: x[0]))
def test_magnet_square_zdir(mszdir_case_item: tuple[str, RawReadMSUnit[RawReadMSZdirAnswer]]):
    """Test the z_dir_magnetic_square_core function."""

    case_name, msdir_case = mszdir_case_item
    msdir_case_answer = msdir_case["ans"]
    msdir_case_counts = msdir_case["counts"]

    assert (
        len(msdir_case_counts) == 1
    ), "The counts should be a single item for the z_dir_magnetic_square_core function."

    py_result = z_dir_magnetic_square_core(
        shots=msdir_case_answer["shots"],
        single_counts=msdir_case_counts[0],
        num_qubits=msdir_case_answer["num_qubits"],
        backend="Python",
    )
    rust_result = z_dir_magnetic_square_core(
        shots=msdir_case_answer["shots"],
        single_counts=msdir_case_counts[0],
        num_qubits=msdir_case_answer["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, str, float]] = [
        ("py", "Python", py_result[0]),
        ("rust", "Rust", rust_result[0]),
        ("ans", "Answer", msdir_case_answer["magnet_square"]),
    ]
    for (name_01, desc_01, result_01), (name_02, desc_02, result_02) in combinations(
        comparison_target, 2
    ):
        assert np.abs(result_01 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{desc_01} and {desc_02} results are not equal in entangled_entropy_core: "
            + f"{name_01}: {result_01}, {name_02}: {result_02} - "
            + f"{name_01}:, {name_02}:"
        )
    for name_01, desc_01, result_01 in comparison_target:
        assert np.abs(result_01 - ANSWERS[case_name]) < ANSWERS_ERROR, (
            f"Result by {desc_01} {result_01} is not close to expected "
            f"{ANSWERS[case_name]} in error {ANSWERS_ERROR}."
        )


@pytest.mark.parametrize("ms_case_item", sorted(raw_ms_case.items(), key=lambda x: x[0]))
def test_magnet_square(ms_case_item: tuple[str, RawReadMSUnit[RawReadMSAnswer]]):
    """Test the z_dir_magnetic_square_core function."""

    case_name, ms_case = ms_case_item
    ms_case_answer = ms_case["ans"]
    ms_case_counts = ms_case["counts"]

    predict_counts_num = ms_case_answer["num_qubits"] * (ms_case_answer["num_qubits"] - 1)
    assert len(ms_case_counts) == predict_counts_num, (
        f"The counts should have {predict_counts_num} items, "
        f"but got {len(ms_case_counts)} for {case_name}"
    )

    py_result = magnetic_square_core(
        shots=ms_case_answer["shots"],
        counts=ms_case_counts,
        num_qubits=ms_case_answer["num_qubits"],
        backend="Python",
    )
    rust_result = magnetic_square_core(
        shots=ms_case_answer["shots"],
        counts=ms_case_counts,
        num_qubits=ms_case_answer["num_qubits"],
        backend="Rust",
    )

    comparison_target: list[tuple[str, str, float]] = [
        ("py", "Python", py_result[0]),
        ("rust", "Rust", rust_result[0]),
        ("ans", "Answer", ms_case_answer["magnet_square"]),
    ]
    for (name_01, desc_01, result_01), (name_02, desc_02, result_02) in combinations(
        comparison_target, 2
    ):
        assert np.abs(result_01 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{desc_01} and {desc_02} results are not equal in entangled_entropy_core: "
            + f"{name_01}: {result_01}, {name_02}: {result_02} - "
            + f"{name_01}:, {name_02}:"
        )
    for name_01, desc_01, result_01 in comparison_target:
        assert np.abs(result_01 - ANSWERS[case_name]) < ANSWERS_ERROR, (
            f"Result by {desc_01} {result_01} is not close to expected "
            f"{ANSWERS[case_name]} in error {ANSWERS_ERROR}."
        )
