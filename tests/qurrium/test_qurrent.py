"""Test the qurry.qurrent module EntropyMeasure class.

- hadamard test at shots = 1024
    - [4-trivial] 0.0 <= 0.25. 1.0 ~= 1.0
    - [4-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5
    - [4-topological-period] 0.033203125 <= 0.25. 0.283203125 ~= 0.25
    - [6-trivial] 0.0 <= 0.25. 1.0 ~= 1.0
    - [6-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5
    - [6-topological-period] 0.041015625 <= 0.25. 0.291015625 ~= 0.25

- randomized measurement and randomized measurement v1 at N_U = 20, shots = 1024
    - [4-trivial] 1.1271525859832763 <= 0.25. 1.1034276962280274 ~= 1.0
    - [4-GHZ] 0.14542131423950194 <= 0.25. 0.35457868576049806 ~= 0.5
    - [4-topological-period] 0.003579425811767567 <= 0.25. 0.25357942581176757 ~= 0.25
    - [6-trivial] 0.18802957534790044 <= 0.25. 0.8119704246520996 ~= 1.0
    - [6-GHZ] 0.018079471588134777 <= 0.25. 0.4819205284118652 ~= 0.5
    - [6-topological-period] 0.003579425811767567 <= 0.25. 0.25357942581176757 ~= 0.25

- randomized measurement at N_U = 50, shots = 1024 with dynamic CNOT gate
    - [4-entangle-by-dyn] 0.035245056152343866 <= 0.25. 1.0352450561523439 ~= 1.0
    - [4-entangle-by-dyn-half] 0.0016211700439453525 <= 0.25. 0.5016211700439454 ~= 0.5
    - [4-dummy-2-body-with-clbits] 0.171049690246582 <= 0.25. 0.828950309753418 ~= 1.0
    - [6-entangle-by-dyn] 0.171562385559082 <= 0.25. 1.171562385559082 ~= 1.0
    - [6-entangle-by-dyn-half] 0.0015624618530273304 <= 0.25. 0.5015624618530273 ~= 0.5
    - [6-dummy-2-body-with-clbits] 0.04613777160644528 <= 0.25. 1.0461377716064453 ~= 1.0

"""

from typing import Any
import os
import pytest
import numpy as np

from qiskit import QuantumCircuit

from utils import (
    current_time_filename,
    InputUnitTuple,
    ResultUnitDict,
    quantity_units_conclusion,
    multi_output_all_conclusion,
    specific_analysis_args_making,
    check_unit,
    detect_simulator_source,
    prepare_random_unitary_seeds,
    item_name_making,
)
from circuits import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits

from qurry.qurrent import EntropyMeasure
from qurry.qurrium import QurriumPrototype
from qurry.tools.backend.import_simulator import GeneralSimulator
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet


SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_unitary_seeds = prepare_random_unitary_seeds()
SIM_DEFAULT_SOURCE = detect_simulator_source()

input_items: dict[str, list[InputUnitTuple]] = {
    "01": [],
    "02": [],
    "03": [],
    "02_extra_clbits": [],
}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {
    "01": [],
    "02": [],
    "03": [],
    "02_extra_clbits": [],
    "01_multi": [],
    "02_multi": [],
    "03_multi": [],
    "02_extra_clbits_multi": [],
}
"""Result items. """

circuits: dict[str, QuantumCircuit] = {
    "4-trivial": TrivialParamagnet(4),
    "4-GHZ": GHZ(4),
    "4-topological-period": TopologicalParamagnet(4),
    "6-trivial": TrivialParamagnet(6),
    "6-GHZ": GHZ(6),
    "6-topological-period": TopologicalParamagnet(6),
    # extra qubits
    "4-dummy-2-body-with-clbits": DummyTwoBodyWithDedicatedClbits(4),
    "6-dummy-2-body-with-clbits": DummyTwoBodyWithDedicatedClbits(6),
    # dynamic circuit
    "4-entangle-by-dyn": CNOTDynCase4To8(4),
    "6-entangle-by-dyn": CNOTDynCase4To8(6),
    "4-entangle-by-dyn-comparison": CNOTDynCase4To8(4, export="comparison"),
    "6-entangle-by-dyn-comparison": CNOTDynCase4To8(6, export="comparison"),
}
"""Circuits. """

# hadamard test/randomized measurement/randomized measurement v1
exp_method_01 = EntropyMeasure(method="hadamard")
exp_method_02 = EntropyMeasure(method="randomized")
exp_method_03 = EntropyMeasure(method="randomized_v1")


def make_01_item(circ_name: str, answer: float) -> InputUnitTuple:
    """Make the item for the first test item division of hadamard test.

    Args:
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input unit tuple for the first test item division.
    """

    return InputUnitTuple(
        ("hadamard", circ_name), {"wave": circ_name, "degree": (0, 2)}, {}, answer
    )


def make_02_item(times: int, num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make the item for the second test item division of randomized measurement.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input unit tuple for the second test item division.
    """

    return InputUnitTuple(
        ("randomized", circ_name),
        {
            "wave": circ_name,
            "times": times,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
        },
        {"selected_qubits": range(-2, 0)},
        answer,
    )


def make_03_item(times: int, num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make the item for the third test item division of randomized measurement v1.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input unit tuple for the third test item division.
    """

    return InputUnitTuple(
        ("randomized_v1", circ_name),
        {
            "wave": circ_name,
            "times": times,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
        },
        {"degree": (0, 2)},
        answer,
    )


for num_qubits_tmp, circ_name_tmp, answer_tmp in [
    (4, "4-trivial", 1.0),
    (4, "4-GHZ", 0.5),
    (4, "4-topological-period", 0.25),
    (6, "6-trivial", 1.0),
    (6, "6-GHZ", 0.5),
    (6, "6-topological-period", 0.25),
]:
    # hadamard test
    input_items["01"].append(make_01_item(circ_name_tmp, answer_tmp))
    exp_method_01.add(circuits[circ_name_tmp], circ_name_tmp)
    # randomized measurement
    input_items["02"].append(make_02_item(20, num_qubits_tmp, circ_name_tmp, answer_tmp))
    exp_method_02.add(circuits[circ_name_tmp], circ_name_tmp)
    # randomized measurement v1
    input_items["03"].append(make_03_item(20, num_qubits_tmp, circ_name_tmp, answer_tmp))
    exp_method_03.add(circuits[circ_name_tmp], circ_name_tmp)

exp_method_02_extra_clbits = EntropyMeasure(method="randomized")


def make_02_extra_clbits_item(
    times: int, num_qubits: int, measure_range: list[int], circ_name: str, answer: float
) -> InputUnitTuple:
    """Make the item for the second test item division of randomized measurement with extra clbits.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        measure_range (list[int]): The range of qubits to measure.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input unit tuple for the second test item division.
    """

    return InputUnitTuple(
        ("randomized_extra_clbits", circ_name),
        {
            "wave": circ_name,
            "times": times,
            "measure": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
        },
        {"selected_qubits": measure_range},
        answer,
    )


for num_qubits_tmp, measure_range_tmp, circ_name_tmp, answer_tmp in [
    (4, [2, 3], "4-dummy-2-body-with-clbits", 1.0),
    (6, [4, 5], "6-dummy-2-body-with-clbits", 1.0),
] + (
    [
        (4, [0, 3], "4-entangle-by-dyn", 1.0),
        (4, [0], "4-entangle-by-dyn", 0.5),
        (6, [0, 5], "6-entangle-by-dyn", 1.0),
        (6, [0], "6-entangle-by-dyn", 0.5),
    ]
    if SIM_DEFAULT_SOURCE == "qiskit_aer"
    else []
):
    input_items["02_extra_clbits"].append(
        make_02_extra_clbits_item(50, num_qubits_tmp, measure_range_tmp, circ_name_tmp, answer_tmp)
    )
    exp_method_02_extra_clbits.add(circuits[circ_name_tmp], circ_name_tmp)


def other_quantities_names(test_item_division: str) -> list[str]:
    """Get other quantities names.

    Args:
        test_item_division (str):
            The test item division.

    Returns:
        list[str]: The other quantities names.
    """
    if test_item_division == "01":
        return ["entropy"]
    if test_item_division == "03":
        return ["entropy", "purityAllSys", "entropyAllSys", "allSystemSource"]
    return ["entropy", "purityAllSys", "entropyAllSys", "all_system_source"]


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion(
        [
            (exp_method_01, "01"),
            (exp_method_02, "02"),
            (exp_method_03, "03"),
            (exp_method_02_extra_clbits, "02_extra_clbits"),
        ],
        input_items,
    ),
)
def test_quantity_unit(
    exp_method: QurriumPrototype, division: str, input_item: InputUnitTuple
) -> None:
    """Test the quantity .

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        input_item (InputUnitTuple): The input item containing measure, analyze, and answer.
    """

    exp_id = exp_method.measure(**input_item.measure, backend=backend)  # type: ignore
    analysis_01 = exp_method.exps[exp_id].analyze(**input_item.analyze)
    quantity_01 = analysis_01.content._asdict()

    if division != "01":
        analysis_02 = exp_method.exps[exp_id].analyze(
            **input_item.analyze, counts_used=range(5)  # type: ignore
        )
        quantity_02 = analysis_02.content._asdict()

        analysis_03 = exp_method.exps[exp_id].analyze(
            **input_item.analyze, counts_used=range(5)  # type: ignore
        )
        quantity_03 = analysis_03.content._asdict()

        all_system_source_keyname = "allSystemSource" if division == "03" else "all_system_source"

        assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
            "The all system entropy should be different for counts_used is not same: "
            + f"counts_used: '{quantity_01['counts_used']}' and '{quantity_02['counts_used']}'."
            + f"'{quantity_01['entropyAllSys']}' != '{quantity_02['entropyAllSys']}', "
            + f"from '{quantity_01[all_system_source_keyname]}' "
            + f"and '{quantity_02[all_system_source_keyname]}'."
        )

        assert (
            np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"])
            < NUMERICAL_ERROR_TOLERANCE
        ), (
            "The all system entropy should be the same for same all system source: "
            + f"{quantity_03['entropyAllSys']} == {quantity_02['entropyAllSys']}."
            + f"from {quantity_03[all_system_source_keyname]} "
            + f"and {quantity_02[all_system_source_keyname]}."
        )

        assert (
            quantity_02[all_system_source_keyname] == "independent"
        ), f"The source of all system is not independent: {quantity_02[all_system_source_keyname]}."

    result_items[division].append(
        check_unit(
            quantity_01,
            "purity",
            input_item.answer,
            input_item.item_name,
            THREDHOLD,
            other_quantities_names(division),
        )
    )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion(
        [
            (exp_method_01, "01", "qurrent_hadamard"),
            (exp_method_02, "02", "qurrent_randomized"),
            (exp_method_03, "03", "qurrent_randomized_v1"),
            (exp_method_02_extra_clbits, "02_extra_clbits", "qurrent_randomized_extra_clbits"),
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
        config_list,
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        skip_output_write=summoner_name != "qurrent_hadamard",
        multiprocess_build=True,
    )

    if summoner_name == "qurrent_hadamard":
        report_001 = exp_method.multimanagers[summoner_id].quantity_container["auto_report"]
    else:
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
                    "purity",
                    answer_dict[config["tags"]],
                    item_name_making(*config["tags"]),
                    THREDHOLD,
                    other_quantities_names(division),
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
        f"results_qurrent.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
