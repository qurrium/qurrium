"""Test the qurry.qurrium module SamplingExecuter and WavesExecuter."""

import os
import pytest

from qiskit import QuantumCircuit

from utils import wave_loader, InputUnit, ResultUnit

from qurry.qurrium import SamplingExecuter, WavesExecuter
from qurry.qurrium.qurrium import QurriumPrototype
from qurry.tools.backend import GeneralSimulator
from qurry.recipe import GHZ, TopologicalParamagnet, TrivialParamagnet

SEED_SIMULATOR = 2019  # <harmony/>

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore

test_items: dict[str, dict[str, InputUnit]] = {}
"""Test items. """
result_items: dict[str, dict[str, ResultUnit]] = {}
"""Result items. """

circuits: dict[str, QuantumCircuit] = {
    "4-trivial": TrivialParamagnet(4),
    "4-GHZ": GHZ(4),
    "4-topological-period": TopologicalParamagnet(4),
    "6-trivial": TrivialParamagnet(6),
    "6-GHZ": GHZ(6),
    "6-topological-period": TopologicalParamagnet(6),
}
"""Circuits. """


exp_demo_01 = SamplingExecuter()
test_items["01"] = {
    circ_name: {
        "measure": {
            "wave": circ_name,
            "sampling": 5,
        },
        "analyze": {},
        "answer": 0,
    }
    for num_qubits, circ_name, answer in [
        (4, "4-trivial", 1.0),
        (4, "4-GHZ", 0.5),
        (4, "4-topological-period", 0.25),
    ]
}
wave_loader(
    exp_demo_01,
    [(circ_name, circuits[circ_name]) for circ_name in test_items["01"]],
)


exp_demo_02 = WavesExecuter()
test_items["02"] = {
    circ_name: {
        "measure": {
            "waves": [circ_name for _ in range(5)],
        },
        "analyze": {},
        "answer": answer,
    }
    for num_qubits, circ_name, answer in [
        (4, "4-trivial", 1.0),
        (4, "4-GHZ", 0.5),
        (4, "4-topological-period", 0.25),
    ]
}
wave_loader(
    exp_demo_02,
    [(circ_name, circuits[circ_name]) for circ_name in test_items["02"]],
)


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
    exp_method: QurriumPrototype,
    test_item_division: str,
    test_item_name: str,
    test_item: InputUnit,
) -> None:
    """Test the quantity of echo.

    Args:
        exp_method (QurriumPrototype):
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


@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "summoner_name"],
    [
        (exp_demo_01, "01", "sampling_excuter"),
        (exp_demo_02, "02", "waves_excuter"),
    ],
)
def test_multi_output_all(
    exp_method: QurriumPrototype,
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

    config_list, analysis_args, answer_list, test_item_name_list = [], [], [], []
    for test_item_name, test_item in list(test_items[test_item_division].items())[:2]:
        config_list.append(test_item["measure"])
        analysis_args.append(test_item["analyze"])
        answer_list.append(test_item["answer"])
        test_item_name_list.append(test_item_name)

    summoner_id = exp_method.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        multiprocess_build=True,
    )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."
