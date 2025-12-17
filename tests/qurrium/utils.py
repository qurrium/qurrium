"""Utility functions for testing qurry package."""

import os
from typing import TypedDict, Any, Optional, Iterable, NamedTuple, Literal, Union
import warnings
import numpy as np

from qurry.qurrium import QurriumPrototype
from qurry.capsule import quickRead
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.tools.datetime import current_time
from qurry.tools.backend.import_simulator import SIM_DEFAULT_SOURCE, SIMULATOR_SOURCES
from qurry.exceptions import QurryDependenciesNotWorking

SEED_FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")
BASIS_FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_basis.json")


def detect_simulator_source() -> str:
    """Detect the simulator source.
    If the default simulator source is not Qiskit Aer, a warning is raised.
    This function is used to check if the Qiskit Aer simulator is available.

    Returns:
        str: The simulator source.
    """

    if SIM_DEFAULT_SOURCE != "qiskit_aer":
        warnings.warn(
            f"Qiskit Aer is not used as the default simulator: {SIM_DEFAULT_SOURCE}. "
            f"Current simulator source is: {SIMULATOR_SOURCES[SIM_DEFAULT_SOURCE]},"
            "some test cases may be skipped.",
            category=QurryDependenciesNotWorking,
        )
    return SIM_DEFAULT_SOURCE


def prepare_random_unitary_seeds(
    filename: str = SEED_FILE_LOCATION,
) -> dict[int, dict[int, dict[int, Union[Literal[0, 1, 2], int]]]]:
    """Prepare random unitary seeds from a file.

    Args:
        filename (str): The filename containing the random unitary seeds.

    Returns:
        dict[str, dict[str, dict[str, int]]]: The random unitary seeds.
    """

    random_unitary_seeds_raw: dict[str, dict[str, dict[str, int]]] = quickRead(filename)
    random_unitary_seeds = {
        int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
        for k, v in random_unitary_seeds_raw.items()
    }
    return random_unitary_seeds


def prepare_random_basis(
    filename: str = BASIS_FILE_LOCATION,
) -> dict[int, dict[int, dict[int, int]]]:
    """Prepare random basis from a file.

    Args:
        filename (str): The filename containing the random basis.

    Returns:
        dict[int, dict[int, dict[int, int]]]: The random basis.
    """

    random_basis_raw: dict[str, dict[str, dict[str, int]]] = quickRead(filename)
    random_basis = {
        int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
        for k, v in random_basis_raw.items()
    }
    return random_basis


def current_time_filename():
    """Returns the current time as a filename.

    Returns:
        str: The current time as a filename.
    """
    return current_time().replace(":", "").replace("-", "").replace(" ", "_")


class InputUnitTuple(NamedTuple):
    """Test unit.

    This is a tuple containing:
    - item_name: str: The name of the test item.
    - measure_draft: dict[str, Any]: The measurement input.
    - analyze: dict[str, Any]: The analysis input.
    - answer: float: The expected answer.
    """

    tags: tuple[str, ...]
    """The tags associated with the test item."""
    measure_draft: dict[str, Any]
    """The measurement input draft."""
    analyze: dict[str, Any]
    """The analysis input."""
    answer: float
    """The expected answer."""

    @property
    def measure(self) -> dict[str, Any]:
        """Get the measurement input.

        Returns:
            dict[str, Any]: The measurement input.
        """
        return {**self.measure_draft, "tags": self.tags}

    @property
    def item_name(self) -> str:
        """Get the item name from the tags.

        Returns:
            str: The item name.
        """
        return item_name_making_from_iter(self.tags)


class ResultUnitDict(TypedDict):
    """Result unit."""

    item_name: str
    """The name of the test item."""
    answer: float
    """The answer from the quantity."""
    diff: float
    """The difference between the answer and the target quantity."""
    target_quantity: float
    """The target quantity to compare against."""
    target_quantity_name: str
    """The name of the target quantity."""
    is_correct: bool
    """Whether the answer is correct or not."""


def check_unit(
    quantity: dict[str, Any],
    target_quantity_name: str,
    answer: float,
    test_item_name: str,
    threshold: float = NUMERICAL_ERROR_TOLERANCE,
    other_quantity_names: Optional[list[str]] = None,
) -> ResultUnitDict:
    """Check the unit of the test.

    Args:
        quantity (dict[str, Any]):
            The quantity to check.
        target_quantity_name (str):
            The name of the target quantity.
        answer (float):
            The expected answer.
        test_item_name (str):
            The name of the test item.
        threshold (float, optional):
            The threshold for the check. Default is NUMERICAL_ERROR_TOLERANCE.
        other_quantity_names (Optional[list[str]]):
            Other quantities to check.

    Returns:
        ResultUnit: The result of the check.
    """

    assert all(
        [target_quantity_name in quantity]
        + ([k in quantity for k in other_quantity_names] if other_quantity_names else [])
    ), (
        f"{test_item_name} | The necessary quantities '{target_quantity_name}' "
        + (f"or other quantities: {other_quantity_names}" if other_quantity_names else "")
        + f" not found in quantity. Quantity: {quantity}"
    )

    diff = np.abs(quantity[target_quantity_name] - answer)
    is_correct = diff < threshold
    assert is_correct, (
        f"{test_item_name} | The result of '{target_quantity_name}' is not correct: "
        + f"{diff} !< {threshold}, {quantity[target_quantity_name]} != {answer}."
    )

    return ResultUnitDict(
        item_name=test_item_name,
        answer=quantity[target_quantity_name],
        diff=diff,
        target_quantity=answer,
        target_quantity_name=target_quantity_name,
        is_correct=is_correct,
    )


def item_name_making_from_iter(iterable: Iterable[str]) -> str:
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


def item_name_making(*iterable: str) -> str:
    """Make an item name from a variable number of strings.

    Args:
        *iterable (str): The strings to join.

    Returns:
        str: The item name.
    """
    return item_name_making_from_iter(iterable)


def quantity_units_conclusion(
    exp_method_and_division_list: list[tuple[QurriumPrototype, str]],
    input_items: dict[str, list[InputUnitTuple]],
) -> list[tuple[QurriumPrototype, str, InputUnitTuple]]:
    """Create a list of quantity units for testing.

    Args:
        exp_method_and_division_list (list[tuple[QurriumPrototype, str]]):
            The list of experiment methods and divisions.
        input_items (dict[str, list[InputUnitTuple]]):
            The input items for each division.

    Returns:
        list[tuple[QurriumPrototype, str, InputUnitTuple]]:
            The list of quantity units.
    """
    return [
        (exp_method, division, input_item)
        for exp_method, division in exp_method_and_division_list
        for input_item in input_items[division]
    ]


def multi_output_all_conclusion(
    exp_method_division_summoner_list: list[tuple[QurriumPrototype, str, str]],
    input_items: dict[str, list[InputUnitTuple]],
) -> list[
    tuple[
        QurriumPrototype,
        str,
        str,
        list[dict[str, Any]],
        dict[tuple[str, ...], dict[str, Any]],
        dict[tuple[str, ...], float],
    ]
]:
    """Create a list of multi-output all conclusions.

    Args:
        exp_method_division_summoner_list (list[tuple[QurriumPrototype, str, str]]):
            The list of experiment methods, divisions, and summoner names.
        input_items (dict[str, list[InputUnitTuple]]):
            The input items for each division.

    Returns:
        list[tuple[QurriumPrototype, str, list[dict[str, Any]], str]]:
            The list of multi-output all conclusions. Each tuple contains:

            - QurriumPrototype: The experiment method.
            - str: The division.
            - str: The summoner name.
            - list[dict[str, Any]]: The configuration list.
            - dict[tuple[str, ...], dict[str, Any]]: The analysis arguments.
            - dict[tuple[str, ...], float]: The answer dictionary.
    """

    result_list = []
    for exp_method, division, summoner_name in exp_method_division_summoner_list:
        config_list, analysis_args, answer_dict = [], {}, {}
        for input_item in input_items[division]:
            config_list.append(input_item.measure)
            analysis_args[input_item.tags] = input_item.analyze
            answer_dict[input_item.tags] = input_item.answer

        result_list.append(
            (
                exp_method,
                division,
                summoner_name,
                config_list,
                analysis_args,
                answer_dict,
            )
        )
    return result_list


def specific_analysis_args_making(
    exp_method: QurriumPrototype,
    summoner_id: str,
    analysis_args: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
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
        exp_id: analysis_args[config["tags"]]
        for exp_id, config in exp_method.multimanagers[summoner_id].beforewards.exps_config.items()
    }
