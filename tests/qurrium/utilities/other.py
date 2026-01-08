"""Miscellaneous utilities for testing. (:mod:`utilities.other`)"""

from typing import TypedDict, Any, Optional, Iterable, NamedTuple, Generic, Union, cast, TypeVar
import os
from pathlib import Path
import numpy as np

from qiskit import QuantumCircuit

from qurry.qurrium import QurriumPrototype
from qurry.qurrium.container import _MA
from qurry.qurrium.analysis import _RA, AnalysisResultsPrototype
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.tools.datetime import current_time

FloatType = Union[float, np.float64]
"""The type alias for :class:`float` and :class:`~numpy.float64`."""


def get_test_export_dir() -> Path:
    """Get the export directory for test outputs.

    Returns:
        Path: The export directory path
    """

    current_dir = os.path.dirname(__file__)
    tests_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if os.path.basename(os.path.abspath(tests_root)) != "tests":
        raise RuntimeError("The tests root directory could not be determined.")
    return Path(tests_root) / "exports"


EXPORT_DIR = get_test_export_dir()
"""The export directory for test outputs."""


def make_current_time_str():
    """Make the current time string for filenames.

    Returns:
        str: The current time string.
    """
    return current_time().replace(":", "").replace("-", "").replace(" ", "_")


class CaseDataDict(TypedDict):
    """Case data dictionary for testing."""

    circuit: QuantumCircuit
    """The quantum circuit to be tested."""
    expect_answer: float
    """The expected answer for the test case."""


class CaseEntriesTuple(NamedTuple, Generic[_MA, _RA]):
    """The case entries tuple for testing."""

    tags: tuple[str, ...]
    """The tags associated with the test item."""
    measure_entries: _MA
    """The measurement input draft."""
    analyze_entries: _RA
    """The analysis input."""
    expect_answer: float
    """The expected answer."""

    def measure_entries_with_tags(self, *more_tags: str) -> _MA:
        """Get the measurement input.

        Args:
            more_tags (str):
                Additional tags to include.

        Returns:
            _MA: The measurement input.
        """
        return cast(_MA, {**self.measure_entries, "tags": self.tags + more_tags})

    @property
    def name(self) -> str:
        """Get the item name from the tags.

        Returns:
            str: The item name.
        """
        return tags_to_name(self.tags)


class AnalysisResultCheckReport(NamedTuple):
    """The analysis result check report."""

    name: str
    """The name of the test item."""
    target_field: str
    """The name of the target field."""
    got_answer: float
    """The answer from the quantity."""
    expect_answer: float
    """The expected answer to compare against."""
    diff: float
    """The difference between the answer and the expected answer."""
    threshold: float
    """The threshold for the check."""
    is_correct: bool
    """Whether the answer is correct or not."""

    def make_logger(self) -> str:
        """Make logger string for the report.

        Returns:
            str: The logger string.
        """
        status = "PASS" if self.is_correct else "FAIL"
        return (
            f"{self.name} | {self.target_field} | {status} | "
            + f"Got: {self.got_answer}, Expect: {self.expect_answer}, "
            + f"Diff: {self.diff} < Threshold: {self.threshold}"
        )


def check_analysis_result(
    result: AnalysisResultsPrototype,
    target_field: str,
    expect_answer: float,
    name: str,
    threshold: float = NUMERICAL_ERROR_TOLERANCE,
    other_fields: Optional[list[str]] = None,
) -> AnalysisResultCheckReport:
    """Check the analysis result for a specific field.

    Args:
        result (AnalysisResultsPrototype): The analysis result to check.
        target_field (str): The name of the target field to check.
        expect_answer (float): The expected answer to compare against.
        name (str): The name of the test item.
        threshold (float, optional): The threshold for the check.
            Defaults to NUMERICAL_ERROR_TOLERANCE.
        other_fields (Optional[list[str]], optional):
            Other fields to check for existence. Defaults to None.

    Returns:
        AnalysisResultCheckReport: The report of the analysis result check.
    """

    assert target_field in result.fields, (
        f"{name} | The necessary quantities '{target_field}' "
        + f" not found in quantity. Quantity: {result.fields}"
    )
    if other_fields is not None:
        assert all(k in result.fields for k in other_fields), (
            f"{name} | The other fields '{other_fields}' "
            + f" not found in quantity. Quantity: {result.fields}"
        )

    diff = np.abs(getattr(result, target_field) - expect_answer)
    is_correct = diff < threshold
    assert is_correct, (
        f"{name} | The result of '{target_field}' is not correct: "
        + f"{diff} !< {threshold}, {getattr(result, target_field)} != {expect_answer}."
    )

    return AnalysisResultCheckReport(
        name=name,
        target_field=target_field,
        got_answer=float(getattr(result, target_field)),
        expect_answer=expect_answer,
        diff=float(diff),
        threshold=threshold,
        is_correct=is_correct,
    )


def tags_to_name(iterable: Iterable[str]) -> str:
    """Make an item name from an iterable of strings.

    Args:
        iterable (Iterable[str]): The iterable of strings.

    Returns:
        str: The item name.
    """
    item_name = ".".join(iterable)
    if item_name:
        return item_name
    raise ValueError("The iterable is empty, cannot create an item name.")


_CET = TypeVar("_CET", bound=CaseEntriesTuple)
"""The type variable for CaseEntriesTuple."""


def make_specific_analysis_args(
    exp_method: QurriumPrototype,
    summoner_id: str,
    analysis_entries_dict: dict[tuple[str, ...], _CET],
):
    """Create specific analysis arguments for a given experiment method and summoner ID.

    Args:
        exp_method (QurriumPrototype): The experiment method.
        summoner_id (str): The ID of the summoner.
        analysis_args (dict[tuple[str, ...], dict[str, Any]]): The analysis arguments.

    Returns:
        dict[str, dict[str, Any]]:
            A dictionary mapping experiment IDs to their specific analysis arguments.
    """

    return {
        exp_id: analysis_entries_dict[config["tags"]].analyze_entries
        for exp_id, config in exp_method.multimanagers[summoner_id].beforewards.exps_config.items()
    }
