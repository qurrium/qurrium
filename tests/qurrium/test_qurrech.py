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

import os
import warnings
import pytest

from qiskit import QuantumCircuit

from utils import current_time_filename, InputUnit, ResultUnit, check_unit
from circuits import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits, ghz_overlap_case

from qurry.qurrech import EchoListen
from qurry.qurrium.qurrium import QurriumPrototype
from qurry.tools.backend.import_simulator import (
    SIM_DEFAULT_SOURCE,
    SIM_IMPORT_ERROR_INFOS,
    GeneralSimulator,
)
from qurry.capsule import quickRead, quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet
from qurry.exceptions import QurryDependenciesNotWorking


SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.26

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore

SEED_FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")
random_unitary_seeds_raw: dict[str, dict[str, dict[str, int]]] = quickRead(SEED_FILE_LOCATION)
random_unitary_seeds = {
    int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
    for k, v in random_unitary_seeds_raw.items()
}


if SIM_DEFAULT_SOURCE != "qiskit_aer":
    warnings.warn(
        f"Qiskit Aer is not used as the default simulator: {SIM_DEFAULT_SOURCE}. "
        f"Please check the simulator source: {SIM_IMPORT_ERROR_INFOS}.",
        category=QurryDependenciesNotWorking,
    )

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

# hadamard test
exp_method_01 = EchoListen(method="hadamard")
test_items["01"] = {}
for circ_name, answer in [
    ("4-trivial", 1.0),
    ("4-GHZ", 0.5),
    ("4-topological-period", 0.25),
    ("6-trivial", 1.0),
    ("6-GHZ", 0.5),
    ("6-topological-period", 0.25),
]:
    test_items["01"][".".join(("hadamard", circ_name, circ_name))] = {
        "measure": {
            "wave1": circ_name,
            "wave2": circ_name,
            "degree": (0, 2),
            "tags": ("hadamard", circ_name, circ_name),
        },
        "analyze": {},
        "answer": answer,
    }
    exp_method_01.add(circuits[circ_name], circ_name)

# randomized measurement
exp_method_02 = EchoListen(method="randomized")
test_items["02"] = {}
for num_qubits, circ_name, answer in [
    (4, "4-trivial", 1.0),
    (4, "4-GHZ", 0.5),
    (4, "4-topological-period", 0.25),
    (6, "6-trivial", 1.0),
    (6, "6-GHZ", 0.5),
    (6, "6-topological-period", 0.25),
]:
    test_items["02"][".".join(("randomized", circ_name, circ_name))] = {
        "measure": {
            "wave1": circ_name,
            "wave2": circ_name,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(20)},
            "tags": ("randomized", circ_name, circ_name),
        },
        "analyze": {"selected_classical_registers": range(-2, 0)},
        "answer": answer,
    }
    exp_method_02.add(circuits[circ_name], circ_name)

# randomized measurement v1
exp_method_03 = EchoListen(method="randomized_v1")
test_items["03"] = {}
for num_qubits, circ_name, answer in [
    (4, "4-trivial", 1.0),
    (4, "4-GHZ", 0.5),
    (4, "4-topological-period", 0.25),
    (6, "6-trivial", 1.0),
    (6, "6-GHZ", 0.5),
    (6, "6-topological-period", 0.25),
]:
    test_items["03"][".".join(("randomized_v1", circ_name, circ_name))] = {
        "measure": {
            "wave1": circ_name,
            "wave2": circ_name,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(20)},
            "tags": ("randomized_v1", circ_name, circ_name),
        },
        "analyze": {"degree": (0, 2)},
        "answer": answer,
    }
    exp_method_03.add(circuits[circ_name], circ_name)

exp_method_02_extra_clbits = EchoListen(method="randomized")
test_items["02_extra_clbits"] = {}
for num_qubits, measure_range, circ_name, answer in [
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
    test_items["02_extra_clbits"][".".join(("randomized_extra_clbits", circ_name, circ_name))] = {
        "measure": {
            "wave1": circ_name,
            "wave2": circ_name,
            "times": 50,
            "measure_1": measure_range,
            "measure_2": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(50)},
            "tags": ("randomized_extra_clbits", circ_name, circ_name),
        },
        "analyze": {"selected_classical_registers": measure_range},
        "answer": answer,
    }
    exp_method_02_extra_clbits.add(circuits[circ_name], circ_name)

exp_method_02_true_overlap = EchoListen(method="randomized")
test_items["02_true_overlap"] = {}
for num_qubits, measure_range, circ_name_1, circ_name_2, selected_cregs, answer in [
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
    test_items["02_true_overlap"][
        ".".join(("randomized_true_overlap", circ_name_1, circ_name_2))
    ] = {
        "measure": {
            "wave1": circ_name_1,
            "wave2": circ_name_2,
            "times": 50,
            "measure_1": measure_range,
            "measure_2": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(50)},
            "tags": ("randomized_true_overlap", circ_name_1, circ_name_2),
        },
        "analyze": {"selected_classical_registers": selected_cregs},
        "answer": answer,
    }
    exp_method_02_true_overlap.add(circuits[circ_name_1], circ_name_1)
    exp_method_02_true_overlap.add(circuits[circ_name_2], circ_name_2)


test_quantity_unit_targets = []
"""Test quantity unit targets.
"""
for exp_method_tmp, test_item_division_tmp in [
    (exp_method_01, "01"),
    (exp_method_02, "02"),
    (exp_method_03, "03"),
    (exp_method_02_extra_clbits, "02_extra_clbits"),
    (exp_method_02_true_overlap, "02_true_overlap"),
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
    exp_method.exps[exp_id].analyze(**test_item["analyze"])

    quantity = exp_method.exps[exp_id].reports[0].content._asdict()

    if test_item_division not in result_items:
        result_items[test_item_division] = {}
    result_items[test_item_division][test_item_name] = check_unit(
        quantity,
        "echo",
        test_item["answer"],
        THREDHOLD,
        test_item_name,
    )


@pytest.mark.dependency(depends=["test_quantity_unit"])
@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "summoner_name"],
    [
        (exp_method_01, "01", "qurrech_hadamard"),
        (exp_method_02, "02", "qurrech_randomized"),
        (exp_method_03, "03", "qurrech_randomized_v1"),
        (exp_method_02_extra_clbits, "02_extra_clbits", "qurrech_randomized_extra_clbits"),
        (exp_method_02_true_overlap, "02_true_overlap", "qurrech_randomized_true_overlap"),
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
        config_list,
        backend=backend,
        summoner_name=summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        skip_build_write=True,
        skip_output_write=True,
        multiprocess_build=True,
    )

    specific_analysis_args = {
        exp_id: analysis_args[".".join(config["tags"])]
        for exp_id, config in exp_method.multimanagers[summoner_id].beforewards.exps_config.items()
    }

    summoner_id = exp_method.multiAnalysis(
        summoner_id,
        specific_analysis_args=specific_analysis_args,  # type: ignore
    )

    for rk, report in exp_method.multimanagers[summoner_id].quantity_container.items():
        for config in config_list:
            for quantity in report[config["tags"]]:
                assert isinstance(quantity, dict), (
                    f"The quantity is not a dict: {quantity}, "
                    + f"{quantity.keys()}/{".".join(config["tags"])}/{rk}."
                )

                if f"{test_item_division}_multi" not in result_items:
                    result_items[f"{test_item_division}_multi"] = {}

                result_items[f"{test_item_division}_multi"][".".join(config["tags"])] = check_unit(
                    quantity,
                    "echo",
                    answer_dict[".".join(config["tags"])],
                    THREDHOLD,
                    ".".join(config["tags"]),
                )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


@pytest.mark.dependency(depends=["test_multi_output_all"])
def test_export():
    """Export the results."""

    quickJSON(
        result_items,
        f"results_qurrech.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
