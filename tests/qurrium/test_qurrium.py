"""Test the qurry.qurrium module SamplingExecuter and WavesExecuter."""

import os
from typing import Union
import pytest

from qiskit import QuantumCircuit

from utils import InputUnit, ResultUnit, check_unit

from qurry.qurrium import SamplingExecuter, WavesExecuter
from qurry.tools.backend import GeneralSimulator
from qurry.recipe import GHZ, TopologicalParamagnet, TrivialParamagnet

SEED_SIMULATOR = 2019  # <harmony/>

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore

test_items: dict[str, dict[str, InputUnit]] = {
    "01": {},
    "02": {},
}
"""Input items. """
result_items: dict[str, dict[str, ResultUnit]] = {
    "01": {},
    "02": {},
    "01_multi.report.001": {},
    "02_multi.report.001": {},
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
test_items["01"] = {}
for num_qubits, circ_name in [
    (4, "4-trivial"),
    (4, "4-GHZ"),
    (4, "4-topological-period"),
]:
    test_items["01"][".".join(("sampling_excuter", circ_name))] = {
        "measure": {"wave": circ_name, "sampling": 5, "tags": ("sampling_excuter", circ_name)},
        "analyze": {},
        "answer": 42,
    }
    exp_demo_01.add(circuits_with_measure[circ_name], circ_name)


exp_demo_02 = WavesExecuter()
test_items["02"] = {}
for num_qubits, circ_name in [
    (4, "4-trivial"),
    (4, "4-GHZ"),
    (4, "4-topological-period"),
]:
    test_items["02"][".".join(("waves_excuter", circ_name))] = {
        "measure": {
            "waves": [circ_name for _ in range(5)],
            "tags": ("waves_excuter", circ_name),
        },
        "analyze": {},
        "answer": 42,
    }
    exp_demo_02.add(circuits_with_measure[circ_name], circ_name)


test_quantity_unit_targets = []
"""Test quantity unit targets.
"""
for exp_method_tmp, test_item_division_tmp in [
    (exp_demo_01, "01"),
    (exp_demo_02, "02"),
]:
    for test_item_name_tmp, test_item_tmp in test_items[test_item_division_tmp].items():
        test_quantity_unit_targets.append(
            (exp_method_tmp, test_item_division_tmp, test_item_name_tmp, test_item_tmp)
        )


@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "test_item_name", "test_item"],
    test_quantity_unit_targets,
)
def test_quantity_unit(
    exp_method: Union[SamplingExecuter, WavesExecuter],
    test_item_division: str,
    test_item_name: str,
    test_item: InputUnit,
) -> None:
    """Test the quantity of echo.

    Args:
        exp_method (Union[SamplingExecuter, WavesExecuter]):
            The QurriumPrototype instance.
        test_item_division (str):
            The test item division.
        test_item_name (str):
            The name of the test item.
        test_item (TestUnit):
            The test item.
    """

    exp_id = exp_method.measure(**test_item["measure"], backend=backend)  # type: ignore

    if test_item_division == "01":
        assert isinstance(exp_method, SamplingExecuter), "The exp_method is not SamplingExecuter."
        assert exp_method.exps[exp_id].args.sampling == test_item["measure"]["sampling"], (
            "The sampling is wrong: "
            + f"{exp_method.exps[exp_id].args.sampling} != {test_item['measure']['sampling']}, "
            + f"on {test_item_name}."
        )
    else:
        assert isinstance(exp_method, WavesExecuter), "The exp_method is not WavesExecuter."
        assert len(exp_method.exps[exp_id].beforewards.circuit) == len(
            test_item["measure"]["waves"]
        ), (
            "The number of waves is wrong: "
            + f"{len(exp_method.exps[exp_id].beforewards.circuit)} != "
            + f"{len(test_item['measure']['waves'])}, {test_item_name}."
        )

    exp_method.exps[exp_id].analyze(**test_item["analyze"])

    quantity = exp_method.exps[exp_id].reports[0].content._asdict()

    result_items[test_item_division][test_item_name] = check_unit(
        quantity,
        "ultimate_answer",
        test_item["answer"],
        1e-12,
        test_item_name,
    )


@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "summoner_name"],
    [
        (exp_demo_01, "01", "sampling_excuter"),
        (exp_demo_02, "02", "waves_excuter"),
    ],
)
def test_multi_output_all(
    exp_method: Union[SamplingExecuter, WavesExecuter],
    test_item_division: str,
    summoner_name: str,
) -> None:
    """Test the multi-output of echo.

    Args:
        exp_method (QurriumPrototype):
            The QurriumPrototype instance.
        test_item_division (str):
            The test item division.
        summoner_name (str):
            The summoner name.
    """

    config_list, analysis_args, answer_dict = [], {}, {}
    for test_item_name, test_item in test_items[test_item_division].items():
        config_list.append(test_item["measure"])
        analysis_args[test_item_name] = test_item["analyze"]
        answer_dict[test_item_name] = test_item["answer"]
        assert test_item_name == ".".join(test_item["measure"]["tags"]), (
            "The test item name is not equal to the tags: "
            + f"{test_item_name} != {'.'.join(test_item['measure']['tags'])}"
        )

    summoner_id = exp_method.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        multiprocess_write=True,
        multiprocess_build=True,
    )

    specific_analysis_args = {
        exp_id: analysis_args[".".join(config["tags"])]
        for exp_id, config in exp_method.multimanagers[summoner_id].beforewards.exps_config.items()
    }

    summoner_id = exp_method.multiAnalysis(
        summoner_id,
        analysis_name="report",
        specific_analysis_args=specific_analysis_args,  # type: ignore
    )
    report_001 = exp_method.multimanagers[summoner_id].quantity_container["report.001"]

    for config in config_list:
        for quantity in report_001[config["tags"]]:
            assert isinstance(quantity, dict), (
                f"The quantity is not a dict: {quantity}, "
                + f"{quantity.keys()}/{'.'.join(config['tags'])}/report.001."
            )

            result_items[f"{test_item_division}_multi.report.001"][".".join(config["tags"])] = (
                check_unit(
                    quantity,
                    "ultimate_answer",
                    42,
                    1e-12,
                    ".".join(config["tags"]),
                )
            )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."
