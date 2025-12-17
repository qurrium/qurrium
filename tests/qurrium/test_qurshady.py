"""Test the qurry.qurrent module ShadowUnveil class.

- classical shadow at N_U = 400, shots = 1

- classical shadow at N_U = 400, shots = 1 with dynamic CNOT gate

"""

import os
from typing import Any
from itertools import combinations
import pytest
import numpy as np

from qiskit import QuantumCircuit

from utils import (
    current_time_filename,
    InputUnitTuple,
    ResultUnitDict,
    multi_output_all_conclusion,
    quantity_units_conclusion,
    specific_analysis_args_making,
    check_unit,
    detect_simulator_source,
    prepare_random_basis,
    item_name_making,
)
from circuits import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits

from qurry.qurrent import ShadowUnveil
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.classical_shadow import (
    JAX_AVAILABLE,
    set_cpu_only,
    RhoMethod,
    TraceMethod,
    purity_value_kind,
    PurityValueKind,
)
from qurry.tools.backend.import_simulator import GeneralSimulator
from qurry.capsule import quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet

set_cpu_only()

SEED_SIMULATOR = 2019  # <harmony/>
THRESHOLD = 0.25
SNAPSHOTS = 400
SHOTS = 4

backend = GeneralSimulator()
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore
random_bases = prepare_random_basis()
SIM_DEFAULT_SOURCE = detect_simulator_source()

methods_by_kind: dict[PurityValueKind, list[tuple[str, str]]] = {}

for rho_method_tmp in RhoMethod.get_all_methods():
    for trace_method_tmp in TraceMethod.get_all_methods():
        if not JAX_AVAILABLE and trace_method_tmp == TraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX.value:
            continue
        methods_by_kind.setdefault(purity_value_kind(rho_method_tmp, trace_method_tmp), []).append(
            (rho_method_tmp, trace_method_tmp)
        )


input_items: dict[str, list[InputUnitTuple]] = {"04": [], "04_extra_clbits": []}
"""Input items. """
result_items: dict[str, list[ResultUnitDict]] = {}
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

exp_method_04 = ShadowUnveil()


def make_04_item(num_qubits: int, circ_name: str, answer: float) -> InputUnitTuple:
    """Make an input item for the fourth experiment.

    Args:
        snapshots (int): The number of measurement times.
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("classical_shadow", circ_name),
        {
            "wave": circ_name,
            "shots": SHOTS,
            "snapshots": SNAPSHOTS,
            "random_basis": {i: random_bases[num_qubits][i] for i in range(SNAPSHOTS)},
        },
        {"selected_qubits": range(-2, 0)},
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
    input_items["04"].append(make_04_item(num_qubits_tmp, circ_name_tmp, answer_tmp))
    exp_method_04.add(circuits[circ_name_tmp], circ_name_tmp)


exp_method_04_extra_clbits = ShadowUnveil()


def make_04_extra_clbits_item(
    num_qubits: int, circ_name: str, measure_range: list[int], answer: float
) -> InputUnitTuple:
    """Make an input item for the fourth experiment with extra clbits.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        circ_name (str): The name of the circuit.
        measure_range (list[int]): The range of measurement.
        answer (float): The expected answer.

    Returns:
        InputUnitTuple: The input item.
    """
    return InputUnitTuple(
        ("classical_shadow_extra_clbits", circ_name, "-".join(map(str, measure_range))),
        {
            "wave": circ_name,
            "measure": measure_range,
            "shots": SHOTS,
            "snapshots": SNAPSHOTS,
            "random_basis": {i: random_bases[num_qubits][i] for i in range(SNAPSHOTS)},
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
    input_items["04_extra_clbits"].append(
        make_04_extra_clbits_item(num_qubits_tmp, circ_name_tmp, measure_range_tmp, answer_tmp)
    )
    exp_method_04_extra_clbits.add(circuits[circ_name_tmp], circ_name_tmp)


@pytest.mark.parametrize(
    ["exp_method", "division", "input_item"],
    quantity_units_conclusion(
        [
            (exp_method_04, "04"),
            (exp_method_04_extra_clbits, "04_extra_clbits"),
        ],
        input_items,
    ),
)
def test_quantity_unit(exp_method: ShadowUnveil, division: str, input_item: InputUnitTuple) -> None:
    """Test the quantity.

    Args:
        exp_method (QurriumPrototype): The QurriumPrototype instance.
        division (str): The test item division.
        input_item (InputUnitTuple): The input item containing measure, analyze, and answer.
    """

    exp_id = exp_method.measure(**input_item.measure, backend=backend)  # type: ignore

    quantity: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for kind, all_methods in methods_by_kind.items():
        for rho_method, trace_method in all_methods:
            analysis_01_tmp = exp_method.exps[exp_id].analyze(
                **input_item.analyze, rho_method=rho_method, trace_method=trace_method
            )
            quantity.setdefault(kind, {})[
                (rho_method, trace_method)
            ] = analysis_01_tmp.content._asdict()

        for (rho_method, trace_method), quantity_item in quantity[kind].items():
            result_items.setdefault(division + f".{rho_method}.{trace_method}", []).append(
                check_unit(
                    quantity_item,
                    "purity",
                    input_item.answer,
                    input_item.item_name,
                    THRESHOLD,
                    # ["entropy", "purityAllSys", "entropyAllSys", "all_system_source"],
                    ["entropy", "mean_of_rho"],
                )
            )
            tmp_mean_of_rho_trace = np.trace(quantity_item["mean_of_rho"])
            assert np.abs(tmp_mean_of_rho_trace - 1) < NUMERICAL_ERROR_TOLERANCE, (
                "The trace of the mean_of_rho should be 1, but error larger than tolerance: "
                + f"{NUMERICAL_ERROR_TOLERANCE}, the trace: {tmp_mean_of_rho_trace}."
            )

        for (rho_trace_1, quantity_item_1), (rho_trace_2, quantity_item_2) in combinations(
            quantity[kind].items(), 2
        ):
            assert (
                np.abs(quantity_item_1["purity"] - quantity_item_2["purity"])
                < NUMERICAL_ERROR_TOLERANCE
            ), (
                "The purity should be the same for same rho and trace method: "
                + f"{rho_trace_1} != {rho_trace_2}: "
                + f"{quantity_item_1['purity']} != {quantity_item_2['purity']}."
            )


@pytest.mark.parametrize(
    ["exp_method", "division", "summoner_name", "config_list", "analysis_args", "answer_dict"],
    multi_output_all_conclusion(
        [
            (exp_method_04, "04", "qurshady"),
            (exp_method_04_extra_clbits, "04_extra_clbits", "qurshady_extra_clbits"),
        ],
        input_items,
    ),
)
def test_multi_output_all(
    exp_method: ShadowUnveil,
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
        # skip_build_write=True,
        # skip_output_write=True,
        multiprocess_build=True,
    )

    specific_analysis_args = specific_analysis_args_making(exp_method, summoner_id, analysis_args)

    methods = sum(methods_by_kind.values(), [])
    for i, (rho_method, trace_method) in enumerate(methods):
        print(f"Processing {i+1}/{len(methods)}: {rho_method}, {trace_method}...")
        summoner_id = exp_method.multiAnalysis(
            summoner_id,
            specific_analysis_args=specific_analysis_args,  # type: ignore
            rho_method=rho_method,
            trace_method=trace_method,
            no_serialize=True,
            analysis_name=f"multi_process.{rho_method}.{trace_method}",
            skip_write=i < len(methods) - 1,  # only write the last one
            multiprocess_analysis=True,
            multiprocess_write=True,
        )

    for rk, report in exp_method.multimanagers[summoner_id].quantity_container.items():
        for config in config_list:
            for quantity in report[config["tags"]]:
                assert isinstance(quantity, dict), (
                    f"The quantity is not a dict: {quantity}, "
                    + f"{quantity.keys()}/{config['tags']}/{rk}."
                )

                result_items.setdefault(f"{division}_multi.{rk}", []).append(
                    (
                        check_unit(
                            quantity,
                            "purity",
                            answer_dict[config["tags"]],
                            item_name_making(*config["tags"]),
                            THRESHOLD,
                            # ["entropy", "purityAllSys", "entropyAllSys", "all_system_source"],
                            ["entropy", "mean_of_rho"],
                        )
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
        f"results_qurshady.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
