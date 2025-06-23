"""Test qurry.process.string_operator module."""

from typing import TypedDict, Literal
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.string_operator import string_operator_availability
from qurry.process.string_operator.strop_core import string_operator_core

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "strop-case.json")


class RawReadAnswer(TypedDict):
    """TypedDict for magnet square answer from JSON."""

    order: float
    num_qubits: int
    shots: int
    i: int
    k: int
    length: int
    str_op: Literal["i", "zy"]
    on_dir: Literal["x", "y"]


class RawReadUnit(TypedDict):
    """TypedDict for magnet square unit from JSON."""

    counts: list[dict[str, int]]
    answer: RawReadAnswer
    tags: list[str]


ANSWERS = {
    "i": {
        "trivial_5": 1.0,
        "trivial_6": 1.0,
        "trivial_7": 1.0,
        "trivial_8": 1.0,
        "trivial_9": 1.0,
        "topological_6": 0.0,
        "topological_8": 0.0,
    },
    "zy": {
        "trivial_7": 0.0,
        "trivial_8": 0.0,
        "trivial_9": 0.0,
        "topological_8": 1.0,
    },
}

ANSWERS_ERROR = 0.05


raw_cases: list[RawReadUnit] = quickRead(FILE_LOCATION)


def unfold_and_validate_case(
    case: RawReadUnit,
) -> tuple[list[str], RawReadAnswer, list[dict[str, int]]]:
    """Unfold the case and validate it."""
    assert isinstance(case, dict), "The case should be a dictionary."
    assert "tags" in case, "The case should have 'tags' key."
    assert "answer" in case, "The case should have 'answer' key."
    assert "counts" in case, "The case should have 'counts' key."

    tags = case["tags"]
    answer = case["answer"]
    counts = case["counts"]

    assert isinstance(tags, list), "'tags' should be a list."
    assert len(tags) == 3, "'tags' should have exactly 3 elements."
    assert (
        tags[2].replace("circ=", "") in ANSWERS[tags[0]]
    ), f"Tag {tags[2]} is not in the expected answers for {tags[0]}."
    tags = tags[:2] + [tags[2].replace("circ=", "")]
    assert isinstance(answer, dict), "'answer' should be a dictionary."
    assert isinstance(counts, list), "'counts' should be a list of dictionaries."

    return tags, answer, counts


cases = [unfold_and_validate_case(case) for case in raw_cases]


@pytest.mark.parametrize(["case_tags", "answer", "counts"], cases)
def test_string_operator(case_tags: list[str], answer: RawReadAnswer, counts: list[dict[str, int]]):
    """Test the string_operator_core function."""

    assert string_operator_availability[1][
        "Rust"
    ], f"Rust is not available. Check the error: {string_operator_availability[1]}"

    assert (
        len(counts) == 1
    ), "The counts should be a single item for the string_operator_core function."

    py_result = string_operator_core(
        shots=answer["shots"],
        counts=counts,
        backend="Python",
    )
    rust_result = string_operator_core(
        shots=answer["shots"],
        counts=counts,
        backend="Rust",
    )

    comparison_target: list[tuple[str, float]] = [
        ("Python", py_result),
        ("Rust", rust_result),
        ("Answer", answer["order"]),
    ]
    for (name_1, result_1), (name_02, result_02) in combinations(comparison_target, 2):
        assert np.abs(result_1 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_02} results are not equal in string_operator_core: "
            f"{name_1}: {result_1}, {name_02}: {result_02}."
        )
    for name_1, result_1 in comparison_target:
        assert np.abs(result_1 - ANSWERS[answer["str_op"]][case_tags[-1]]) < ANSWERS_ERROR, (
            f"Result by {name_1} {result_1} is not close to expected "
            f"{ANSWERS[answer["str_op"]][case_tags[-1]]} in error {ANSWERS_ERROR}."
        )
