"""Test the qurry.qurries module SamplingExecuter and WavesExecuter."""

from typing import Union, Any
import os
import pytest

from qiskit import QuantumCircuit

from utils import (
    InputUnitTuple,
    ResultUnitDict,
    quantity_units_conclusion,
    multi_output_all_conclusion,
    specific_analysis_args_making,
    check_unit,
    item_name_making,
)

from qurry.qurries import SamplingExecuter, WavesExecuter
from qurry.tools.backend import GeneralSimulator
from qurry.recipe import GHZ, TopologicalParamagnet, TrivialParamagnet

SEED_SIMULATOR = 2019  # <harmony/>

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore

input_items: dict[str, list[InputUnitTuple]] = {
    "01": [],
    "02": [],
}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {
    "01": [],
    "02": [],
    "01_multi": [],
    "02_multi": [],
}
"""Result items. """

circuits_with_measure: dict[str, QuantumCircuit] = {
    "4-trivial": TrivialParamagnet(4),
    "4-GHZ": GHZ(4),
    "4-topological-period": TopologicalParamagnet(4),
}
"""Circuits. """
for qc in circuits_with_measure.values():
    qc.measure_all()


exp_demo_01 = SamplingExecuter()
exp_demo_02 = WavesExecuter()


def make_01_item(circ_name: str, measure_sampling: int = 5) -> InputUnitTuple:
    """Make an input item for the first experiment.

    Args:
        circ_name (str): The name of the circuit.
        measure_sampling (int): The sampling number.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("sampling_excuter", circ_name),
        {"wave": circ_name, "sampling": measure_sampling},
        {},
        42,
    )


def make_02_item(circ_name: str, measure_waves: int = 5) -> InputUnitTuple:
    """Make an input item for the second experiment.

    Args:
        circ_name (str): The name of the circuit.
        measure_waves (int): The number of waves.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("waves_excuter", circ_name),
        {"waves": [circ_name for _ in range(measure_waves)], "tags": ("waves_excuter", circ_name)},
        {},
        42,
    )


for num_qubits_tmp, circ_name_tmp in [
    (4, "4-trivial"),
    (4, "4-GHZ"),
    (4, "4-topological-period"),
]:
    input_items["01"].append(make_01_item(circ_name_tmp))
    exp_demo_01.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)

    input_items["02"].append(make_02_item(circ_name_tmp))
    exp_demo_02.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion(
        [
            (exp_demo_01, "01"),
            (exp_demo_02, "02"),
        ],
        input_items,
    ),
)
def test_quantity_unit(
    exp_method: Union[SamplingExecuter, WavesExecuter], division: str, input_item: InputUnitTuple
) -> None:
    """Test the quantity of echo.

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        input_item (InputUnitTuple): The input item containing measure, analyze, and answer.
    """

    exp_id = exp_method.measure(**input_item.measure, backend=backend)

    if division == "01":
        assert isinstance(exp_method, SamplingExecuter), "The exp_method is not SamplingExecuter."
        assert exp_method.exps[exp_id].args.sampling == input_item.measure["sampling"], (
            "The sampling is wrong: "
            f"{exp_method.exps[exp_id].args.sampling} != {input_item.measure['sampling']}, "
            f"on {input_item.item_name}."
        )
    else:
        assert isinstance(exp_method, WavesExecuter), "The exp_method is not WavesExecuter."
        assert len(exp_method.exps[exp_id].beforewards.circuit) == len(
            input_item.measure["waves"]
        ), (
            "The number of waves is wrong: "
            f"{len(exp_method.exps[exp_id].beforewards.circuit)} != "
            f"{len(input_item.measure['waves'])}, {input_item.item_name}."
        )

    exp_method.exps[exp_id].analyze(**input_item.analyze)

    quantity = exp_method.exps[exp_id].reports[0].content._asdict()

    result_items[division].append(
        check_unit(
            quantity,
            "ultimate_answer",
            input_item.answer,
            input_item.item_name,
        )
    )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion(
        [
            (exp_demo_01, "01", "qurries_sampling_executer"),
            (exp_demo_02, "02", "qurries_wave_executer"),
        ],
        input_items,
    ),
)
def test_multi_output_all(
    exp_method: Union[SamplingExecuter, WavesExecuter],
    division: str,
    summoner_name: str,
    config_list: list[dict[str, Any]],
    analysis_args: dict[tuple[str, ...], dict[str, Any]],
    answer_dict: dict[tuple[str, ...], float],
) -> None:
    """Test the multi-output of echo.

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        summoner_name (str): The name of the summoner.
        config_list (list[dict[str, Any]]): The configuration list.
        analysis_args (dict[tuple[str, ...], dict[str, Any]]): The analysis arguments.
        answer_dict (dict[tuple[str, ...], float]): The answer dictionary.
    """

    summoner_id = exp_method.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        multiprocess_write=True,
        multiprocess_build=True,
    )

    tmp_analysis_name = "report"
    summoner_id = exp_method.multiAnalysis(
        summoner_id,
        analysis_name=tmp_analysis_name,
        no_serialize=True,
        specific_analysis_args=specific_analysis_args_making(
            exp_method, summoner_id, analysis_args
        ),  # type: ignore
    )
    report_001 = exp_method.multimanagers[summoner_id].quantity_container[tmp_analysis_name]

    for config in config_list:
        for quantity in report_001[config["tags"]]:
            assert isinstance(
                quantity, dict
            ), f"The quantity is not a dict: {quantity}, {quantity.keys()}/{config['tags']}."

            result_items[f"{division}_multi"].append(
                check_unit(
                    quantity,
                    "ultimate_answer",
                    answer_dict[config["tags"]],
                    item_name_making(*config["tags"]),
                )
            )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."
