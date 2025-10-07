"""Test the qurry.qurries module StringOperator classes."""

from typing import Any, Literal
import os
import pytest

from qiskit import QuantumCircuit


from utils import (
    current_time_filename,
    InputUnitTuple,
    ResultUnitDict,
    quantity_units_conclusion,
    multi_output_all_conclusion,
    prepare_random_unitary_seeds,
    check_unit,
    item_name_making,
)

from qurry.qurries import StringOperator
from qurry.tools.backend import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, TopologicalParamagnet

SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.1

ANSWERS = {
    "i": {
        "5-trivial": 1.0,
        "6-trivial": 1.0,
        "7-trivial": 1.0,
        "8-trivial": 1.0,
        "9-trivial": 1.0,
        "6-topological": 0.0,
        "8-topological": 0.0,
    },
    "zy": {
        "7-trivial_": 0.0,
        "8-trivial": 0.0,
        "9-trivial": 0.0,
        "8-topological": 1.0,
    },
}

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_unitary_seeds = prepare_random_unitary_seeds()

input_items: dict[str, list[InputUnitTuple]] = {
    "01": [],
}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {
    "01": [],
    "01_multi": [],
}
"""Result items. """

circuits_with_measure: dict[str, QuantumCircuit] = {
    "5-trivial": TrivialParamagnet(5),
    "6-trivial": TrivialParamagnet(6),
    "7-trivial": TrivialParamagnet(7),
    "8-trivial": TrivialParamagnet(8),
    "9-trivial": TrivialParamagnet(9),
    "6-topological": TopologicalParamagnet(6),
    "8-topological": TopologicalParamagnet(8),
}
"""Circuits. """

exp_method_01 = StringOperator()


def make_01_item(circ_name: str, str_op: Literal["i", "zy"], answer: float) -> InputUnitTuple:
    """Make an input item for the first experiment.

    Args:
        circ_name (str): The name of the circuit.
        str_op (Literal["i", "zy"]): The string operator to be used in the measurement.
        answer (float): The expected answer for the measurement.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("string_operator", circ_name, str_op), {"wave": circ_name, "str_op": str_op}, {}, answer
    )


cases_list: list[tuple[int, str, float, Literal["i", "zy"]]] = [
    (5, "5-trivial", ANSWERS["i"]["5-trivial"], "i"),
    (6, "6-trivial", ANSWERS["i"]["6-trivial"], "i"),
    (7, "7-trivial", ANSWERS["i"]["7-trivial"], "i"),
    (8, "8-trivial", ANSWERS["i"]["8-trivial"], "i"),
    (9, "9-trivial", ANSWERS["i"]["9-trivial"], "i"),
    (6, "6-topological", ANSWERS["i"]["6-topological"], "i"),
    (8, "8-topological", ANSWERS["i"]["8-topological"], "i"),
    (7, "7-trivial", ANSWERS["zy"]["7-trivial_"], "zy"),
    (8, "8-trivial", ANSWERS["zy"]["8-trivial"], "zy"),
    (9, "9-trivial", ANSWERS["zy"]["9-trivial"], "zy"),
    (8, "8-topological", ANSWERS["zy"]["8-topological"], "zy"),
]
for num_qubits_tmp, circ_name_tmp, answer_tmp, str_op_tmp in cases_list:
    input_items["01"].append(make_01_item(circ_name_tmp, str_op_tmp, answer_tmp))
    exp_method_01.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion([(exp_method_01, "01")], input_items),
)
def test_quantity_unit(
    exp_method: StringOperator, division: str, input_item: InputUnitTuple
) -> None:
    """Test the quantity.

    Args:
        exp_method (StringOperator): The QurriumPrototype instance.
        division (str): The test item division.
        input_item (InputUnitTuple): The input item containing measure, analyze, and answer.
    """

    exp_id = exp_method.measure(**input_item.measure, backend=backend)  # type: ignore
    exp_method.exps[exp_id].analyze(**input_item.analyze)
    quantity = exp_method.exps[exp_id].reports[0].content._asdict()

    result_items[division].append(
        check_unit(
            quantity,
            "order",
            input_item.answer,
            input_item.item_name,
            THREDHOLD,
        )
    )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion([(exp_method_01, "01", "qurstrop")], input_items),
)
def test_multi_output_all(
    exp_method: StringOperator,
    division: str,
    summoner_name: str,
    config_list: list[dict[str, Any]],
    analysis_args: dict[tuple[str, ...], dict[str, Any]],
    answer_dict: dict[tuple[str, ...], float],
) -> None:
    """Test the multi-output.

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        summoner_name (str): The name of the summoner.
        config_list (list[dict[str, Any]]): The configuration list.
        analysis_args (dict[tuple[str, ...], dict[str, Any]]): The analysis arguments.
        answer_dict (dict[tuple[str, ...], float]): The answer dictionary.
    """

    summoner_id = exp_method.multiOutput(
        config_list,
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        multiprocess_build=True,
    )

    report_001 = exp_method.multimanagers[summoner_id].quantity_container["auto_report"]

    for config in config_list:
        for quantity in report_001[config["tags"]]:
            assert isinstance(quantity, dict), (
                f"The quantity is not a dict: {quantity}, {quantity.keys()}/{config['tags']}."
            )

            result_items[f"{division}_multi"].append(
                check_unit(
                    quantity,
                    "order",
                    answer_dict[config["tags"]],
                    item_name_making(*config["tags"]),
                    THREDHOLD,
                )
            )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert read_summoner_id == summoner_id, (
        f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."
    )


def test_export():
    """Export the results."""

    quickJSON(
        result_items,
        f"results_qurstrop.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
