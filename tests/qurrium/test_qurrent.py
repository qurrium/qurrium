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

import os
import pytest
import numpy as np

from qiskit import QuantumCircuit

from utils import (
    current_time_filename,
    InputUnit,
    ResultUnit,
    check_unit,
    detect_simulator_source,
    prepare_random_unitary_seeds,
)
from circuits import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits

from qurry.qurrent import EntropyMeasure
from qurry.qurrium.qurrium import QurriumPrototype
from qurry.tools.backend.import_simulator import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet


SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_unitary_seeds = prepare_random_unitary_seeds()
SIM_DEFAULT_SOURCE = detect_simulator_source()

input_items: dict[str, dict[str, InputUnit]] = {
    "01": {},
    "02": {},
    "03": {},
    "02_extra_clbits": {},
}
"""Input items. """
result_items: dict[str, dict[str, ResultUnit]] = {
    "01": {},
    "02": {},
    "03": {},
    "02_extra_clbits": {},
    "01_multi.report.001": {},
    "02_multi.report.001": {},
    "03_multi.report.001": {},
    "02_extra_clbits_multi.report.001": {},
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

for num_qubits, circ_name, answer in [
    (4, "4-trivial", 1.0),
    (4, "4-GHZ", 0.5),
    (4, "4-topological-period", 0.25),
    (6, "6-trivial", 1.0),
    (6, "6-GHZ", 0.5),
    (6, "6-topological-period", 0.25),
]:
    # hadamard test
    input_items["01"][".".join(("hadamard", circ_name))] = {
        "measure": {"wave": circ_name, "degree": (0, 2), "tags": ("hadamard", circ_name)},
        "analyze": {},
        "answer": answer,
    }
    exp_method_01.add(circuits[circ_name], circ_name)

    # randomized measurement
    input_items["02"][".".join(("randomized", circ_name))] = {
        "measure": {
            "wave": circ_name,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(20)},
            "tags": ("randomized", circ_name),
        },
        "analyze": {"selected_qubits": range(-2, 0)},
        "answer": answer,
    }
    exp_method_02.add(circuits[circ_name], circ_name)

    # randomized measurement v1
    input_items["03"][".".join(("randomized_v1", circ_name))] = {
        "measure": {
            "wave": circ_name,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(20)},
            "tags": ("randomized_v1", circ_name),
        },
        "analyze": {"degree": (0, 2)},
        "answer": answer,
    }
    exp_method_03.add(circuits[circ_name], circ_name)

exp_method_02_extra_clbits = EntropyMeasure(method="randomized")
input_items["02_extra_clbits"] = {}
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
    input_items["02_extra_clbits"][".".join(("randomized_extra_clbits", circ_name))] = {
        "measure": {
            "wave": circ_name,
            "times": 50,
            "measure": measure_range,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(50)},
            "tags": ("randomized_extra_clbits", circ_name),
        },
        "analyze": {"selected_qubits": measure_range},
        "answer": answer,
    }
    exp_method_02_extra_clbits.add(circuits[circ_name], circ_name)

test_quantity_unit_targets = []
"""Test quantity unit targets.
"""
for exp_method_tmp, test_item_division_tmp in [
    (exp_method_01, "01"),
    (exp_method_02, "02"),
    (exp_method_03, "03"),
    (exp_method_02_extra_clbits, "02_extra_clbits"),
]:
    for test_item_name_tmp, test_item_tmp in input_items[test_item_division_tmp].items():
        test_quantity_unit_targets.append(
            (exp_method_tmp, test_item_division_tmp, test_item_name_tmp, test_item_tmp)
        )


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


@pytest.mark.order(1)
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
    analysis_01 = exp_method.exps[exp_id].analyze(**test_item["analyze"])
    quantity_01 = analysis_01.content._asdict()

    if test_item_division != "01":
        analysis_02 = exp_method.exps[exp_id].analyze(
            **test_item["analyze"], counts_used=range(5)  # type: ignore
        )
        quantity_02 = analysis_02.content._asdict()

        analysis_03 = exp_method.exps[exp_id].analyze(
            **test_item["analyze"], counts_used=range(5)  # type: ignore
        )
        quantity_03 = analysis_03.content._asdict()

        all_system_source_keyname = (
            "allSystemSource" if test_item_division == "03" else "all_system_source"
        )

        assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
            "The all system entropy should be different for counts_used is not same: "
            + f"counts_used: {quantity_01['counts_used']} and {quantity_02['counts_used']}."
            + f"{quantity_01['entropyAllSys']} != {quantity_02['entropyAllSys']}, "
            + f"from {quantity_01[all_system_source_keyname]} "
            + f"and {quantity_02[all_system_source_keyname]}."
        )
        assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
            "The all system entropy should be the same for same all system source: "
            + f"{quantity_03['entropyAllSys']} == {quantity_02['entropyAllSys']}."
            + f"from {quantity_03[all_system_source_keyname]} "
            + f"and {quantity_02[all_system_source_keyname]}."
        )
        assert (
            quantity_02[all_system_source_keyname] == "independent"
        ), f"The source of all system is not independent: {quantity_02[all_system_source_keyname]}."
        assert "AnalysisHeader" in quantity_03[all_system_source_keyname], (
            "The source of all system is not from existed analysis: "
            + f"{quantity_03[all_system_source_keyname]}."
        )

    result_items[test_item_division][test_item_name] = check_unit(
        quantity_01,
        "purity",
        test_item["answer"],
        THREDHOLD,
        test_item_name,
        other_quantities_names(test_item_division),
    )


@pytest.mark.order(2)
@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "summoner_name"],
    [
        (exp_method_01, "01", "qurrent_hadamard"),
        (exp_method_02, "02", "qurrent_randomized"),
        (exp_method_03, "03", "qurrent_randomized_v1"),
        (exp_method_02_extra_clbits, "02_extra_clbits", "qurrent_randomized_extra_clbits"),
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
    for test_item_name, test_item in input_items[test_item_division].items():
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
                    "purity",
                    answer_dict[".".join(config["tags"])],
                    THREDHOLD,
                    ".".join(config["tags"]),
                    other_quantities_names(test_item_division),
                )
            )

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


@pytest.mark.order(3)
def test_export():
    """Export the results."""

    quickJSON(
        result_items,
        f"results_qurrent.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
