"""Test the qurry.qurrent module ShadowUnveil class.

- classical shadow at N_U = 100, shots = 1024
    - [4-trivial] 0.1396211346712979 <= 0.25, 0.8603788653287021 ~= 1.0
    - [4-GHZ] 0.020000482039018164 <= 0.25, 0.5200004820390182 ~= 0.5
    - [4-topological-period] 4.4909390534142446e-07 <= 0.25, 0.25000044909390534 ~= 0.25
    - [6-trivial] 0.1880350251631303 <= 0.25, 0.8119649748368697 ~= 1.0
    - [6-GHZ] 0.06589450836181643 <= 0.25, 0.43410549163818357 ~= 0.5
    - [6-topological-period] 8.716583251966448e-06 <= 0.25, 0.25000871658325197 ~= 0.25

- classical shadow at N_U = 100, shots = 1024 with dynamic CNOT gate
    - [4-entangle-by-dyn] 0.24680606321855025 <= 0.25, 0.7531939367814497 ~= 1.0
    - [4-entangle-by-dyn-half] 1.4655373313188225e-05 <= 0.25, 0.4999853446266868 ~= 0.5
    - [4-dummy-2-body-with-clbits] 0.014617952866987749 <= 0.25, 0.9853820471330123 ~= 1.0
    - [6-entangle-by-dyn] 0.1609451276605785 <= 0.25, 0.839054872339422 ~= 1.0
    - [6-entangle-by-dyn-half] 1.0452270507832484e-05 <= 0.25, 0.49998954772949217 ~= 0.5
    - [6-dummy-2-body-with-clbits] 0.09746155305342241 <= 0.25, 0.9025384469465776 ~= 1.0

"""

import os
from typing import Any
from itertools import combinations
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

from qurry.qurrent import ShadowUnveil
from qurry.qurrent.classical_shadow import ShadowUnveilAnalysis
from qurry.process.classical_shadow.matrix_calcution import JAX_AVAILABLE, set_cpu_only
from qurry.tools.backend.import_simulator import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet

set_cpu_only()

SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_unitary_seeds = prepare_random_unitary_seeds()
SIM_DEFAULT_SOURCE = detect_simulator_source()

rho_methods = ["numpy", "numpy_precomputed", "numpy_flatten"] + (
    ["jax_flatten"] if JAX_AVAILABLE else []
)
trace_methods = ["trace_of_matmul", "einsum_ij_ji", "einsum_aij_bji_to_ab_numpy"] + (
    ["einsum_aij_bji_to_ab_jax"] if JAX_AVAILABLE else []
)

input_items: dict[str, dict[str, InputUnit]] = {
    "04": {},
    "04_extra_clbits": {},
}
"""Input items. """
result_items: dict[str, dict[str, ResultUnit]] = {
    f"{items_name}.{rho_method}.{trace_method}": {}
    for items_name in ["04", "04_extra_clbits"]
    for rho_method in rho_methods
    for trace_method in trace_methods
}
"""Result items. """
result_items.update(
    {
        f"{items_name}_multi.{rho_method}.{trace_method}": {}
        for items_name in ["04", "04_extra_clbits"]
        for rho_method in rho_methods
        for trace_method in trace_methods
    }
)

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

exp_method_04 = ShadowUnveil()
for num_qubits, circ_name, answer in [
    (4, "4-trivial", 1.0),
    (4, "4-GHZ", 0.5),
    (4, "4-topological-period", 0.25),
    (6, "6-trivial", 1.0),
    (6, "6-GHZ", 0.5),
    (6, "6-topological-period", 0.25),
]:
    input_items["04"][".".join(("classical_shadow", circ_name))] = {
        "measure": {
            "wave": circ_name,
            "times": 80,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(80)},
            "tags": ("classical_shadow", circ_name),
        },
        "analyze": {"selected_qubits": range(-2, 0)},
        "answer": answer,
    }
    exp_method_04.add(circuits[circ_name], circ_name)


exp_method_04_extra_clbits = ShadowUnveil()
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
    input_items["04_extra_clbits"][".".join(("classical_shadow_extra_clbits", circ_name))] = {
        "measure": {
            "wave": circ_name,
            "measure": measure_range,
            "times": 80,
            "random_unitary_seeds": {i: random_unitary_seeds[num_qubits][i] for i in range(80)},
            "tags": ("classical_shadow_extra_clbits", circ_name),
        },
        "analyze": {"selected_qubits": measure_range},
        "answer": answer,
    }
    exp_method_04_extra_clbits.add(circuits[circ_name], circ_name)

test_quantity_unit_targets = []
"""Test quantity unit targets.
"""
for exp_method_tmp, test_item_division_tmp in [
    (exp_method_04, "04"),
    (exp_method_04_extra_clbits, "04_extra_clbits"),
]:
    for test_item_name_tmp, test_item_tmp in input_items[test_item_division_tmp].items():
        test_quantity_unit_targets.append(
            (exp_method_tmp, test_item_division_tmp, test_item_name_tmp, test_item_tmp)
        )


@pytest.mark.order(1)
@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "test_item_name", "test_item"],
    test_quantity_unit_targets,
)
def test_quantity_unit(
    exp_method: ShadowUnveil,
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
    analysis: dict[str, ShadowUnveilAnalysis] = {}
    quantity: dict[str, dict[str, Any]] = {}

    exp_id = exp_method.measure(**test_item["measure"], backend=backend)  # type: ignore
    for rho_method in rho_methods:
        for trace_method in trace_methods:
            analysis[rho_method + "." + trace_method] = exp_method.exps[exp_id].analyze(
                **test_item["analyze"],  # type: ignore
                rho_method=rho_method,
                trace_method=trace_method,
            )
            quantity[rho_method + "." + trace_method] = analysis[
                rho_method + "." + trace_method
            ].content._asdict()

            # analysis_02_tmp = exp_method.exps[exp_id].analyze(
            #     **test_item["analyze"], counts_used=range(5)  # type: ignore
            # )
            # quantity_02_tmp = analysis_02_tmp.content._asdict()

            # analysis_03_tmp = exp_method.exps[exp_id].analyze(
            #     **test_item["analyze"], counts_used=range(5)  # type: ignore
            # )
            # quantity_03_tmp = analysis_03_tmp.content._asdict()

            # all_system_source_keyname = (
            #     "allSystemSource" if test_item_division == "03" else "all_system_source"
            # )

            # assert (
            #     quantity_02_tmp["entropyAllSys"]
            #     != quantity[rho_method + "." + trace_method]["entropyAllSys"]
            # ), (
            #     "The all system entropy should be different for counts_used is not same: "
            #     + f"counts_used: {quantity[rho_method + "." + trace_method]['counts_used']} "
            #     + f"and {quantity_02_tmp['counts_used']}."
            #     + f"{quantity[rho_method + "." + trace_method]['entropyAllSys']} != "
            #     + f"{quantity_02_tmp['entropyAllSys']}, "
            #     + f"from {quantity[rho_method + "." + trace_method][all_system_source_keyname]} "
            #     + f"and {quantity_02_tmp[all_system_source_keyname]}."
            # )
            # assert (
            #     np.abs(quantity_03_tmp["entropyAllSys"]
            #            - quantity_02_tmp["entropyAllSys"]) < 1e-12
            # ), (
            #     "The all system entropy should be the same for same all system source: "
            #     + f"{quantity_03_tmp['entropyAllSys']} == {quantity_02_tmp['entropyAllSys']}."
            #     + f"from {quantity_03_tmp[all_system_source_keyname]} "
            #     + f"and {quantity_02_tmp[all_system_source_keyname]}."
            # )
            # assert quantity_02_tmp[all_system_source_keyname] == "independent", (
            #     "The source of all system is not independent: "
            #     + f"{quantity_02_tmp[all_system_source_keyname]}."
            # )
            # assert "AnalysisHeader" in quantity_03_tmp[all_system_source_keyname], (
            #     "The source of all system is not from existed analysis: "
            #     + f"{quantity_03_tmp[all_system_source_keyname]}."
            # )

    for rho_trace_method, quantity_item in quantity.items():
        result_items[test_item_division + f".{rho_trace_method}"][test_item_name] = check_unit(
            quantity_item,
            "purity",
            test_item["answer"],
            THREDHOLD,
            test_item_name,
            # ["entropy", "purityAllSys", "entropyAllSys", "all_system_source"],
            ["entropy", "expect_rho"],
        )
        tmp_expect_rho_trace = np.trace(quantity_item["expect_rho"])
        assert np.abs(tmp_expect_rho_trace - 1) < 1e-12, (
            "The trace of the expect_rho should be 1: " + f"{tmp_expect_rho_trace}."
        )
    for (rho_trace_method_1, quantity_item_1), (
        rho_trace_method_2,
        quantity_item_2,
    ) in combinations(quantity.items(), 2):
        assert np.abs(quantity_item_1["purity"] - quantity_item_2["purity"]) < 1e-12, (
            "The purity should be the same for same rho and trace method: "
            + f"{rho_trace_method_1} != {rho_trace_method_2}: "
            + f"{quantity_item_1['purity']} != {quantity_item_2['purity']}."
        )


@pytest.mark.order(2)
@pytest.mark.parametrize(
    ["exp_method", "test_item_division", "summoner_name"],
    [
        (exp_method_04, "04", "qurshady"),
        (exp_method_04_extra_clbits, "04_extra_clbits", "qurshady_extra_clbits"),
    ],
)
def test_multi_output_all(
    exp_method: ShadowUnveil,
    test_item_division: str,
    summoner_name: str,
) -> None:
    """Test the multi-output of echo.

    Args:
        exp_method (ShadowUnveil):
            The ShadowUnveil instance.
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

    for rho_method in rho_methods:
        for trace_method in trace_methods:
            summoner_id = exp_method.multiAnalysis(
                summoner_id,
                specific_analysis_args=specific_analysis_args,  # type: ignore
                rho_method=rho_method,
                trace_method=trace_method,
                no_serialize=True,
                analysis_name=f"multi_process.{rho_method}.{trace_method}",
                skip_write=(rho_methods[-1] != rho_method or trace_methods[-1] != trace_method),
                multiprocess_analysis=True,
                multiprocess_write=True,
            )

    for rk, report in exp_method.multimanagers[summoner_id].quantity_container.items():
        for config in config_list:
            for quantity in report[config["tags"]]:
                assert isinstance(quantity, dict), (
                    f"The quantity is not a dict: {quantity}, "
                    + f"{quantity.keys()}/{'.'.join(config['tags'])}/{rk}."
                )

                result_items[f"{test_item_division}_multi.{rk}"][".".join(config["tags"])] = (
                    check_unit(
                        quantity,
                        "purity",
                        answer_dict[".".join(config["tags"])],
                        THREDHOLD,
                        ".".join(config["tags"]),
                        # ["entropy", "purityAllSys", "entropyAllSys", "all_system_source"],
                        ["entropy", "expect_rho"],
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
        f"results_qurshady.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
