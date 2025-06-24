"""Test the qurry.qurrech module EchoListen class.

- hadamard test
    - [4-trivial] 0.0 <= 0.26, 1.0 ~= 1.0
    - [4-GHZ] 0.005859375 <= 0.26, 0.505859375 ~= 0.5
    - [4-topological-period] 0.033203125 <= 0.26, 0.283203125 ~= 0.25
    - [6-trivial] 0.0 <= 0.26, 1.0 ~= 1.0
    - [6-GHZ] 0.005859375 <= 0.26, 0.505859375 ~= 0.5
    - [6-topological-period] 0.041015625 <= 0.26, 0.291015625 ~= 0.25

- randomized measurement and randomized measurement v1
    - [4-trivial] 0.12715258598327628 <= 0.26, 1.1271525859832763 ~= 1.0
    - [4-GHZ] 0.1428383827209473 <= 0.26, 0.3571616172790527 ~= 0.5
    - [4-topological-period] 0.24956111907958983 <= 0.26, 0.25043888092041017 ~= 0.25
    - [6-trivial] 0.1894473552703857 <= 0.26, 0.8105526447296143 ~= 1.0
    - [6-GHZ] 0.020473003387451172 <= 0.26, 0.47952699661254883 ~= 0.5
    - [6-topological-period] 0.24956111907958983 <= 0.26, 0.25043888092041017 ~= 0.25

- randomized measurement at N_U = 50, shots = 1024 with dynamic CNOT gate
    - [4-entangle-by-dyn] 0.03493137359619136 <= 0.26, 1.0349313735961914 ~= 1.0
    - [4-entangle-by-dyn-half] 0.0004758453369140825 <= 0.26, 0.4995241546630859 ~= 0.5
    - [4-dummy-2-body-with-clbits] 0.17455284118652348 <= 0.26, 0.8254471588134765 ~= 1.0
    - [6-entangle-by-dyn] 0.1653023529052735 <= 0.26, 1.1653023529052735 ~= 1.0
    - [6-entangle-by-dyn-half] 0.0005647659301757924 <= 0.26, 0.4994352340698242 ~= 0.5
    - [6-dummy-2-body-with-clbits] 0.04541765213012705 <= 0.26, 1.045417652130127 ~= 1.0
    - [4-entangle-by-dyn/4-entangle-by-dyn-comparison]
        0.17522192001342773 <= 0.26, 1.1752219200134277 ~= 1.0
    - [6-entangle-by-dyn/6-entangle-by-dyn-comparison]
        0.045955753326416104 <= 0.26, 1.045955753326416 ~= 1.0

"""

from typing import Any, Optional
import os
import pytest

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
from circuits import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits, ghz_overlap_case

from qurry.qurrech import EchoListen
from qurry.qurrium.qurrium import QurriumPrototype
from qurry.tools.backend.import_simulator import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet


SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.26

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_unitary_seeds = prepare_random_unitary_seeds()
SIM_DEFAULT_SOURCE = detect_simulator_source()

input_items: dict[str, list[InputUnitTuple]] = {
    "01": [],
    "02": [],
    "03": [],
    "02_extra_clbits": [],
    "02_true_overlap": [],
}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {
    "01": [],
    "02": [],
    "03": [],
    "02_extra_clbits": [],
    "02_true_overlap": [],
    "01_multi": [],
    "02_multi": [],
    "03_multi": [],
    "02_extra_clbits_multi": [],
    "02_true_overlap_multi": [],
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
    # true overlap
    "4-GHZ-00": ghz_overlap_case(4, "00"),
    "4-GHZ-01": ghz_overlap_case(4, "01"),
    "4-GHZ-10": ghz_overlap_case(4, "10"),
    "4-GHZ-11": ghz_overlap_case(4, "11"),
    "4-GHZ-x-init-GHZ": ghz_overlap_case(4, "x-init-GHZ"),
    "4-GHZ-singlet": ghz_overlap_case(4, "singlet"),
    "4-GHZ-intracell-plus": ghz_overlap_case(4, "intracell-plus"),
}
"""Circuits. """

# hadamard test/randomized measurement/randomized measurement v1
exp_method_01 = EchoListen(method="hadamard")
exp_method_02 = EchoListen(method="randomized")
exp_method_03 = EchoListen(method="randomized_v1")


def make_01_item(circ_name: str, answer: float) -> InputUnitTuple:
    """Make the input item for the hadamard test.

    Args:
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("hadamard", circ_name),
        {"wave1": circ_name, "wave2": circ_name, "degree": (0, 2)},
        {},
        answer,
    )


def make_02_item(times: int, num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make the input item for the randomized measurement.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("randomized", circ_name, circ_name),
        {
            "wave1": circ_name,
            "wave2": circ_name,
            "times": times,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
        },
        {"selected_classical_registers": range(-2, 0)},
        answer,
    )


def make_03_item(times: int, num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make the input item for the randomized measurement v1.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("randomized_v1", circ_name, circ_name),
        {
            "wave1": circ_name,
            "wave2": circ_name,
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


exp_method_02_extra_clbits = EchoListen(method="randomized")


def make_02_extra_clbits_item(
    times: int, num_qubits: int, circ_name: str, measure_range: list[int], answer: float
) -> InputUnitTuple:
    """Make the input item for the randomized measurement with extra classical bits.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        measure_range (list[int]): The range of classical registers to measure.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("randomized_extra_clbits", circ_name, circ_name),
        {
            "wave1": circ_name,
            "wave2": circ_name,
            "times": times,
            "measure_1": measure_range,
            "measure_2": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
            "tags": ("randomized_extra_clbits", circ_name, circ_name),
        },
        {"selected_classical_registers": measure_range},
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
        make_02_extra_clbits_item(50, num_qubits_tmp, circ_name_tmp, measure_range_tmp, answer_tmp)
    )
    exp_method_02_extra_clbits.add(circuits[circ_name_tmp], circ_name_tmp)

exp_method_02_true_overlap = EchoListen(method="randomized")


def make_02_true_overlap_item(
    times: int,
    num_qubits: int,
    measure_range: Optional[list[int]],
    circ_name_1: str,
    circ_name_2: str,
    selected_cregs: list[int],
    answer: float,
) -> InputUnitTuple:
    """Make the input item for the true overlap measurement.

    Args:
        times (int): The group number of random unitary.
        num_qubits (int): The number of qubits in the circuit.
        measure_range (Union[None, list[int]]): The range of classical registers to measure.
        circ_name_1 (str): The name of the first circuit.
        circ_name_2 (str): The name of the second circuit.
        selected_cregs (list[int]): The selected classical registers.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("randomized_true_overlap", circ_name_1, circ_name_2),
        {
            "wave1": circ_name_1,
            "wave2": circ_name_2,
            "times": times,
            "measure_1": measure_range,
            "measure_2": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(times)},
            "tags": ("randomized_true_overlap", circ_name_1, circ_name_2),
        },
        {"selected_classical_registers": selected_cregs},
        answer,
    )


for (
    num_qubits_tmp,
    measure_range_tmp,
    circ_name_1_tmp,
    circ_name_2_tmp,
    selected_cregs_tmp,
    answer_tmp,
) in [
    (4, None, "4-GHZ", "4-GHZ-00", range(4), 0.5),
    (4, None, "4-GHZ", "4-GHZ-01", range(4), 0),
    (4, None, "4-GHZ", "4-GHZ-10", range(4), 0),
    (4, None, "4-GHZ", "4-GHZ-11", range(4), 0.5),
    (4, None, "4-GHZ", "4-GHZ-x-init-GHZ", range(4), 0),
    (4, None, "4-GHZ", "4-GHZ-singlet", range(4), 0),
    (4, None, "4-GHZ", "4-GHZ-intracell-plus", range(4), 0),
] + (
    [
        (4, [0, 3], "4-entangle-by-dyn", "4-entangle-by-dyn-comparison", range(-2, 0), 1.0),
        (6, [0, 5], "6-entangle-by-dyn", "6-entangle-by-dyn-comparison", range(-2, 0), 1.0),
    ]
    if SIM_DEFAULT_SOURCE == "qiskit_aer"
    else []
):
    input_items["02_true_overlap"].append(
        make_02_true_overlap_item(
            80,
            num_qubits_tmp,
            measure_range_tmp,
            circ_name_1_tmp,
            circ_name_2_tmp,
            selected_cregs_tmp,
            answer_tmp,
        )
    )
    exp_method_02_true_overlap.add(circuits[circ_name_1_tmp], circ_name_1_tmp)
    exp_method_02_true_overlap.add(circuits[circ_name_2_tmp], circ_name_2_tmp)


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion(
        [
            (exp_method_01, "01"),
            (exp_method_02, "02"),
            (exp_method_03, "03"),
            (exp_method_02_extra_clbits, "02_extra_clbits"),
            (exp_method_02_true_overlap, "02_true_overlap"),
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

    result_items[division].append(
        check_unit(
            quantity,
            "echo",
            input_item.answer,
            input_item.item_name,
            THREDHOLD,
        )
    )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion(
        [
            (exp_method_01, "01", "qurrech_hadamard"),
            (exp_method_02, "02", "qurrech_randomized"),
            (exp_method_03, "03", "qurrech_randomized_v1"),
            (exp_method_02_extra_clbits, "02_extra_clbits", "qurrech_randomized_extra_clbits"),
            (exp_method_02_true_overlap, "02_true_overlap", "qurrech_randomized_true_overlap"),
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
        skip_output_write=summoner_name != "qurrech_hadamard",
        multiprocess_build=True,
    )

    if summoner_name == "qurrech_hadamard":
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
                    "echo",
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
        f"results_qurrech.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
