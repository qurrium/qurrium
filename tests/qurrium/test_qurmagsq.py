"""Test the qurry.qurries module MagnetSquare and ZDirMagnetSquare classes."""

from typing import Any
import os
import functools as ft
from itertools import permutations
import pytest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate, IGate


from utils import (
    current_time_filename,
    InputUnitTuple,
    ResultUnitDict,
    quantity_units_conclusion,
    multi_output_all_conclusion,
    specific_analysis_args_making,
    prepare_random_basis,
    check_unit,
    item_name_making,
)

from qurry.qurries import ZDirMagnetSquare, MagnetSquare
from qurry.qurrent import ShadowUnveil
from qurry.qurrium import QurriumPrototype
from qurry.tools.backend import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import Cat, TrivialParamagnet

SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25
SNAPSHOTS = 500

ANSWERS = {
    "2-trivial": 1 / 2,
    "4-trivial": 1 / 4,
    "6-trivial": 1 / 6,
    "8-trivial": 1 / 8,
    "2-cat": 1,
    "4-cat": 1,
    "6-cat": 1,
    "8-cat": 1,
}

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_bases = prepare_random_basis()

input_items: dict[str, list[InputUnitTuple]] = {
    "01": [],
    "02": [],
    "03": [],
}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {
    "01": [],
    "02": [],
    "03": [],
    "01_multi": [],
    "02_multi": [],
    "03_multi": [],
}
"""Result items. """

circuits_with_measure: dict[str, QuantumCircuit] = {
    "2-trivial": TrivialParamagnet(2),
    "4-trivial": TrivialParamagnet(4),
    "6-trivial": TrivialParamagnet(6),
    "8-trivial": TrivialParamagnet(8),
    "2-cat": Cat(2),
    "4-cat": Cat(4),
    "6-cat": Cat(6),
    "8-cat": Cat(8),
}
"""Circuits. """

exp_method_01 = ZDirMagnetSquare()
exp_method_02 = MagnetSquare()


def make_01_item(circ_name: str, answer: float) -> InputUnitTuple:
    """Make an input item for the first experiment.

    Args:
        circ_name (str): The name of the circuit.
        answer (float): The expected answer for the measurement.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(("zdir_magnet_square", circ_name), {"wave": circ_name}, {}, answer)


def make_02_item(circ_name: str, answer: float) -> InputUnitTuple:
    """Make an input item for the second experiment.

    Args:
        circ_name (str): The name of the circuit.
        answer (float): The expected answer for the measurement.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("magnet_square", circ_name), {"wave": circ_name, "unitary_operator": "z"}, {}, answer
    )


for num_qubits_tmp, circ_name_tmp, answer_tmp in [
    (2, "2-trivial", ANSWERS["2-trivial"]),
    (4, "4-trivial", ANSWERS["4-trivial"]),
    (6, "6-trivial", ANSWERS["6-trivial"]),
    (8, "8-trivial", ANSWERS["8-trivial"]),
    (2, "2-cat", ANSWERS["2-cat"]),
    (4, "4-cat", ANSWERS["4-cat"]),
    (6, "6-cat", ANSWERS["6-cat"]),
    (8, "8-cat", ANSWERS["8-cat"]),
]:
    # zdir magnet square
    input_items["01"].append(make_01_item(circ_name_tmp, answer_tmp))
    exp_method_01.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)
    # magnet square
    input_items["02"].append(make_02_item(circ_name_tmp, answer_tmp))
    exp_method_02.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)

exp_method_03 = ShadowUnveil()


def operator_preparing(
    num_qubits: int,
) -> list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
    """Prepare the operator for the circuit.

    Args:
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
            A list of numpy arrays representing the operator for each pair of qubits.
    """
    z_gate_matrix = ZGate().to_matrix()
    i_gate_matrix = IGate().to_matrix()

    return [
        ft.reduce(
            np.kron,
            (z_gate_matrix.copy() if i in tgt else i_gate_matrix.copy() for i in range(num_qubits)),
        )
        for tgt in permutations(range(num_qubits), 2)
    ]  # type: ignore[return]


def unveil_magnetization_square(
    estimate_of_given_operators: list[np.complex128], num_qubits: int
) -> np.float64:
    """Processing Classical Shadows post-processing for MagnetSquare.

    Args:
        estimate_of_given_operators (list[np.complex128]): The estimates of the given operators.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        np.float64: The unveiled magnet square value.
    """
    return np.complex128(sum(estimate_of_given_operators) + num_qubits).real / (num_qubits**2)


def make_03_item(snapshots: int, num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make an input item for the third experiment.

    Args:
        snapshots (int): The number of snapshots to run the circuit.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer for the measurement.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("magnet_square", circ_name),
        {
            "wave": circ_name,
            "snapshots": snapshots,
            "shots": 10,
            "random_basis": {i: random_bases[num_qubits][i] for i in range(snapshots)},
        },
        {
            "selected_qubits": range(num_qubits),
            "given_operators": operator_preparing(num_qubits),
        },
        answer,
    )


for num_qubits_tmp, circ_name_tmp, answer_tmp in [
    (4, "4-trivial", ANSWERS["4-trivial"]),
    (4, "4-cat", ANSWERS["4-cat"]),
]:
    # classical shadows
    input_items["03"].append(make_03_item(SNAPSHOTS, num_qubits_tmp, circ_name_tmp, answer_tmp))
    exp_method_03.add(circuits_with_measure[circ_name_tmp], circ_name_tmp)


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion(
        [
            (exp_method_01, "01"),
            (exp_method_02, "02"),
            (exp_method_03, "03"),
        ],
        input_items,
    ),
)
def test_quantity_unit(
    exp_method: QurriumPrototype, division: str, input_item: InputUnitTuple
) -> None:
    """Test the quantity.

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        input_item (InputUnitTuple): The input item containing measure, analyze, and answer.
    """

    exp_id = exp_method.measure(**input_item.measure, backend=backend)  # type: ignore
    exp_method.exps[exp_id].analyze(**input_item.analyze)
    quantity = exp_method.exps[exp_id].reports[0].content._asdict()

    if division == "03":
        another_quantity = {
            "magnet_square": unveil_magnetization_square(
                quantity["estimate_of_given_operators"],
                exp_method.exps[exp_id].reports[0].input.num_qubits,
            ),
            **quantity,
        }
        result_items[division].append(
            check_unit(
                another_quantity,
                "magnet_square",
                input_item.answer,
                input_item.item_name,
                THREDHOLD,
                ["purity", "entropy", "estimate_of_given_operators"],
            )
        )
    else:
        result_items[division].append(
            check_unit(
                quantity,
                "magnet_square",
                input_item.answer,
                input_item.item_name,
                THREDHOLD,
            )
        )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion(
        [
            (exp_method_01, "01", "qurmagsq_zdir"),
            (exp_method_02, "02", "qurmagsq"),
            (exp_method_03, "03", "qurshady"),
        ],
        input_items,
    ),
)
def test_multi_output_all(
    exp_method: QurriumPrototype,
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
        skip_output_write=division == "03",
        multiprocess_build=True,
    )

    if division == "03":
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
    else:
        report_001 = exp_method.multimanagers[summoner_id].quantity_container["auto_report"]

    for config in config_list:
        for quantity in report_001[config["tags"]]:
            assert isinstance(
                quantity, dict
            ), f"The quantity is not a dict: {quantity}, {quantity.keys()}/{config['tags']}."

            if division == "03":
                another_quantity = {
                    "magnet_square": unveil_magnetization_square(
                        quantity["estimate_of_given_operators"],
                        quantity["input"]["num_qubits"],
                    ),
                    **quantity,
                }
                result_items[division].append(
                    check_unit(
                        another_quantity,
                        "magnet_square",
                        answer_dict[config["tags"]],
                        item_name_making(*config["tags"]),
                        THREDHOLD,
                        ["purity", "entropy", "estimate_of_given_operators"],
                    )
                )
            else:
                result_items[f"{division}_multi"].append(
                    check_unit(
                        quantity,
                        "magnet_square",
                        answer_dict[config["tags"]],
                        item_name_making(*config["tags"]),
                        THREDHOLD,
                    )
                )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


def test_export():
    """Export the results."""

    quickJSON(
        result_items,
        f"results_qurmagsq.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
