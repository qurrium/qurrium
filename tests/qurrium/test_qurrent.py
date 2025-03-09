"""
================================================================
Test the qurry.qurrent module EntropyMeasure class.
================================================================

- hadamard test
    - [4-trivial] 0.0 <= 0.25. 1.0 ~= 1.0.
    - [4-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5.
    - [4-topological-period] 0.033203125 <= 0.25. 0.283203125 ~= 0.25.
    - [6-trivial] 0.0 <= 0.25. 1.0 ~= 1.0.
    - [6-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5.
    - [6-topological-period] 0.041015625 <= 0.25. 0.291015625 ~= 0.25.

- randomized measurement and randomized measurement v1
    - [4-trivial] 0.10342769622802739 <= 0.25. 1.1034276962280274 ~= 1.0.
    - [4-GHZ] 0.14542131423950194 <= 0.25. 0.35457868576049806 ~= 0.5.
    - [4-topological-period] 0.003579425811767567 <= 0.25. 0.25357942581176757 ~= 0.25.
    - [6-trivial] 0.18802957534790044 <= 0.25. 0.8119704246520996 ~= 1.0.
    - [6-GHZ] 0.018079471588134777 <= 0.25. 0.4819205284118652 ~= 0.5.
    - [6-topological-period] 0.003579425811767567 <= 0.25. 0.25357942581176757 ~= 0.25.

- randomized measurement with dynamic CNOT gate
    - [4-entangle-by-dyn] 0.0015944480895996316 <= 0.25. 0.5015944480895996 ~= 0.5.
    - [6-entangle-by-dyn] 0.0015944480895996316 <= 0.25. 0.5015944480895996 ~= 0.5.

"""

import os
import pytest
import numpy as np

from utils import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits, current_time_filename

from qiskit_aer import AerSimulator

from qurry.qurrent import EntropyMeasure
from qurry.tools.backend import GeneralSimulator
from qurry.capsule import mori, hoshi, quickRead, quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet

dummy_aer_simulator = AerSimulator()

tag_list = mori.TagList()
statesheet = hoshi.Hoshi()

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")
SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25
MANUAL_ASSERT_ERROR = False

exp_method_01 = EntropyMeasure(method="hadamard")
exp_method_02 = EntropyMeasure(method="randomized")
exp_method_02_with_extra_clbits = EntropyMeasure(method="randomized")
exp_method_03 = EntropyMeasure(method="randomized_v1")

random_unitary_seeds_raw: dict[str, dict[str, dict[str, int]]] = quickRead(FILE_LOCATION)
random_unitary_seeds = {
    int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
    for k, v in random_unitary_seeds_raw.items()
}
seed_usage = {}

wave_adds = {
    "01": [],
    "02": [],
    "03": [],
    "02_with_extra_clbits": [],
}
answer = {}
measure_dyn = {}

results = {}

for i in range(4, 7, 2):
    wave_adds["01"].append(exp_method_01.add(TrivialParamagnet(i), f"{i}-trivial"))
    wave_adds["02"].append(exp_method_02.add(TrivialParamagnet(i), f"{i}-trivial"))
    wave_adds["03"].append(exp_method_03.add(TrivialParamagnet(i), f"{i}-trivial"))
    answer[f"{i}-trivial"] = 1.0
    seed_usage[f"{i}-trivial"] = i
    measure_dyn[f"{i}-trivial"] = {
        "01": (0, 2),
        "02": range(-2, 0),
        "03": (0, 2),
    }
    # purity = 1.0

    wave_adds["01"].append(exp_method_01.add(GHZ(i), f"{i}-GHZ"))
    wave_adds["02"].append(exp_method_02.add(GHZ(i), f"{i}-GHZ"))
    wave_adds["03"].append(exp_method_03.add(GHZ(i), f"{i}-GHZ"))
    answer[f"{i}-GHZ"] = 0.5
    seed_usage[f"{i}-GHZ"] = i
    measure_dyn[f"{i}-GHZ"] = {
        "01": (0, 2),
        "02": range(-2, 0),
        "03": (0, 2),
    }
    # purity = 0.5

    wave_adds["01"].append(
        exp_method_01.add(TopologicalParamagnet(i, "period"), f"{i}-topological-period")
    )
    wave_adds["02"].append(
        exp_method_02.add(TopologicalParamagnet(i, "period"), f"{i}-topological-period")
    )
    wave_adds["03"].append(
        exp_method_03.add(TopologicalParamagnet(i, "period"), f"{i}-topological-period")
    )
    answer[f"{i}-topological-period"] = 0.25
    seed_usage[f"{i}-topological-period"] = i
    measure_dyn[f"{i}-topological-period"] = {
        "01": (0, 2),
        "02": range(-2, 0),
        "03": (0, 2),
    }
    # purity = 0.25

    wave_adds["02_with_extra_clbits"].append(
        exp_method_02_with_extra_clbits.add(CNOTDynCase4To8(i), f"{i}-entangle-by-dyn")
    )
    answer[f"{i}-entangle-by-dyn"] = 1
    seed_usage[f"{i}-entangle-by-dyn"] = i
    measure_dyn[f"{i}-entangle-by-dyn"] = {
        "02_with_extra_clbits": [0, i - 1],
    }
    # purity = 1, de-facto all system when selected qubits is [0, i - 1]

    wave_adds["02_with_extra_clbits"].append(
        exp_method_02_with_extra_clbits.add(CNOTDynCase4To8(i), f"{i}-entangle-by-dyn-half")
    )
    answer[f"{i}-entangle-by-dyn-half"] = 0.5
    seed_usage[f"{i}-entangle-by-dyn-half"] = i
    measure_dyn[f"{i}-entangle-by-dyn-half"] = {
        "02_with_extra_clbits": [0],
    }
    # purity = 0.5, when selected qubits is [0]

    wave_adds["02_with_extra_clbits"].append(
        exp_method_02_with_extra_clbits.add(
            DummyTwoBodyWithDedicatedClbits(i), f"{i}-dummy-2-body-with-clbits"
        )
    )
    answer[f"{i}-dummy-2-body-with-clbits"] = 1.0
    seed_usage[f"{i}-dummy-2-body-with-clbits"] = i
    measure_dyn[f"{i}-dummy-2-body-with-clbits"] = {
        "02_with_extra_clbits": [i - 2, i - 1],
    }
    # purity = 1.0

backend = GeneralSimulator()
# backend = BasicAer.backends()[0]
print(backend.configuration())  # type: ignore
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore


@pytest.mark.parametrize("tgt", wave_adds["01"])
def test_quantity_01(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_01.measure(tgt, measure_dyn[tgt]["01"], backend=backend)
    exp_method_01.exps[exp_id].analyze()
    quantity = exp_method_01.exps[exp_id].reports[0].content._asdict()
    assert all(
        ["entropy" in quantity, "purity" in quantity]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity.keys()}."

    diff = np.abs(quantity["purity"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The hadamard test result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity['purity']} != {answer[tgt]}."
    )
    results[tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "is_correct": is_correct,
        "quantity": quantity,
    }


def test_multi_output_01():
    """Test the multi-output of purity and entropy.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave": k,
            "degree": measure_dyn[k]["01"],
        }
        for k in wave_adds["01"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["01"][:3]]

    summoner_id = exp_method_01.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name="qurrent_hadamard",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_01.multiAnalysis(summoner_id)
    quantity_container = exp_method_01.multimanagers[summoner_id].quantity_container
    for rk, report in quantity_container.items():
        for qk, quantities in report.items():
            for qqi, quantity in enumerate(quantities):
                assert isinstance(quantity, dict), f"The quantity is not a dict: {quantity}."
                assert all(["entropy" in quantity, "purity" in quantity]), (
                    "The necessary quantities 'entropy', 'purity' "
                    + f"are not found: {quantity.keys()}-{qk}-{rk}."
                )
                assert np.abs(quantity["purity"] - answer_list[qqi]) < THREDHOLD, (
                    "The hadamard test result is wrong: "
                    + f"{np.abs(quantity['purity'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['purity']} != {answer_list[qqi]}."
                )

    read_summoner_id = exp_method_01.multiRead(
        summoner_name=exp_method_01.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )

    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


@pytest.mark.parametrize("tgt", wave_adds["02"])
def test_quantity_02(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_02.measure(
        wave=tgt,
        times=20,
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(20)},
        backend=backend,
    )
    analysis_01 = exp_method_02.exps[exp_id].analyze(measure_dyn[tgt]["02"])
    quantity_01 = analysis_01.content._asdict()
    analysis_02 = exp_method_02.exps[exp_id].analyze(measure_dyn[tgt]["02"], counts_used=range(5))
    quantity_02 = analysis_02.content._asdict()
    analysis_03 = exp_method_02.exps[exp_id].analyze(measure_dyn[tgt]["02"], counts_used=range(5))
    quantity_03 = analysis_03.content._asdict()

    assert all(
        ["entropy" in quantity_01, "purity" in quantity_01]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
    assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
        "The all system entropy is not changed: "
        + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
        + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
    )
    assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
        "The all system entropy is not changed: "
        + f"{quantity_03['entropyAllSys']} != {quantity_02['entropyAllSys']}."
    )
    assert (
        quantity_02["all_system_source"] == "independent"
    ), f"The source of all system is not independent: {quantity_02['all_system_source']}."
    assert (
        "AnalysisHeader" in quantity_03["all_system_source"]
    ), f"The source of all system is not from existed analysis: {quantity_03['all_system_source']}."

    diff = np.abs(quantity_01["purity"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity_01['purity']} != {answer[tgt]}."
    )
    results[tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "is_correct": is_correct,
        "quantity": quantity_01,
    }


def test_multi_output_02():
    """Test the multi-output of purity and entropy.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave": k,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[seed_usage[k]][i] for i in range(20)},
        }
        for k in wave_adds["02"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["02"][:3]]

    summoner_id = exp_method_02.multiOutput(
        config_list,  # type: ignore
        shots=4096,
        backend=backend,
        summoner_name="qurrent_randomized",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_02.multiAnalysis(
        summoner_id,
        specific_analysis_args={
            ck: {
                "selected_qubits": measure_dyn[wk]["02"],
            }
            for wk, ck in zip(
                wave_adds["02"][:3],
                exp_method_02.multimanagers[summoner_id].afterwards.allCounts.keys(),
            )
        },
    )
    quantity_container = exp_method_02.multimanagers[summoner_id].quantity_container
    for rk, report in quantity_container.items():
        for qk, quantities in report.items():
            for qqi, quantity in enumerate(quantities):
                assert isinstance(quantity, dict), f"The quantity is not a dict: {quantity}."
                assert all(["entropy" in quantity, "purity" in quantity]), (
                    "The necessary quantities 'entropy', 'purity' "
                    + f"are not found: {quantity.keys()}-{qk}-{rk}."
                )
                assert np.abs(quantity["purity"] - answer_list[qqi]) < THREDHOLD, (
                    "The randomized measurement result is wrong: "
                    + f"{np.abs(quantity['purity'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['purity']} != {answer_list[qqi]}."
                )

    read_summoner_id = exp_method_02.multiRead(
        summoner_name=exp_method_02.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )

    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


@pytest.mark.parametrize("tgt", wave_adds["02_with_extra_clbits"])
def test_quantity_02_with_extra_clbits(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_02_with_extra_clbits.measure(
        wave=tgt,
        times=50,
        measure=measure_dyn[tgt]["02_with_extra_clbits"],
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(50)},
        backend=backend,
    )
    analysis_01 = exp_method_02_with_extra_clbits.exps[exp_id].analyze(
        measure_dyn[tgt]["02_with_extra_clbits"]
    )
    quantity_01 = analysis_01.content._asdict()
    analysis_02 = exp_method_02_with_extra_clbits.exps[exp_id].analyze(
        measure_dyn[tgt]["02_with_extra_clbits"], counts_used=range(5)
    )
    quantity_02 = analysis_02.content._asdict()
    analysis_03 = exp_method_02_with_extra_clbits.exps[exp_id].analyze(
        measure_dyn[tgt]["02_with_extra_clbits"], counts_used=range(5)
    )
    quantity_03 = analysis_03.content._asdict()

    assert all(
        ["entropy" in quantity_01, "purity" in quantity_01]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
    assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
        "The all system entropy is not changed: "
        + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
        + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
    )
    assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
        "The all system entropy is not changed: "
        + f"{quantity_03['entropyAllSys']} != {quantity_02['entropyAllSys']}."
    )
    assert (
        quantity_02["all_system_source"] == "independent"
    ), f"The source of all system is not independent: {quantity_02['all_system_source']}."
    assert (
        "AnalysisHeader" in quantity_03["all_system_source"]
    ), f"The source of all system is not from existed analysis: {quantity_03['all_system_source']}."

    diff = np.abs(quantity_01["purity"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity_01['purity']} != {answer[tgt]}. {analysis_01}"
    )
    results[tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "is_correct": is_correct,
        "quantity": quantity_01,
    }


@pytest.mark.parametrize("tgt", wave_adds["03"])
def test_quantity_03(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_03.measure(
        wave=tgt,
        times=20,
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(20)},
        backend=backend,
    )
    analysis_01 = exp_method_03.exps[exp_id].analyze(measure_dyn[tgt]["03"])
    quantity_01 = analysis_01.content._asdict()
    analysis_02 = exp_method_03.exps[exp_id].analyze(measure_dyn[tgt]["03"], counts_used=range(5))
    quantity_02 = analysis_02.content._asdict()
    analysis_03 = exp_method_03.exps[exp_id].analyze(measure_dyn[tgt]["03"], counts_used=range(5))
    quantity_03 = analysis_03.content._asdict()

    assert all(
        ["entropy" in quantity_01, "purity" in quantity_01]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
    assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
        "The all system entropy is not changed: "
        + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
        + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
    )
    assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
        "The all system entropy is not changed: "
        + f"{quantity_03['entropyAllSys']} != {quantity_02['entropyAllSys']}."
    )
    assert (
        quantity_02["allSystemSource"] == "independent"
    ), f"The source of all system is not independent: {quantity_02['allSystemSource']}."
    assert (
        "AnalysisHeader" in quantity_03["allSystemSource"]
    ), f"The source of all system is not from existed analysis: {quantity_03['allSystemSource']}."

    diff = np.abs(quantity_01["purity"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity_01['purity']} != {answer[tgt]}."
    )
    results[tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "is_correct": is_correct,
        "quantity": quantity_01,
    }


def test_multi_output_03():
    """Test the multi-output of purity and entropy.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave": k,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[seed_usage[k]][i] for i in range(20)},
        }
        for k in wave_adds["03"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["03"][:3]]

    summoner_id = exp_method_03.multiOutput(
        config_list,  # type: ignore
        shots=4096,
        backend=backend,
        summoner_name="qurrent_randomized",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_03.multiAnalysis(
        summoner_id,
        specific_analysis_args={
            ck: {
                "degree": measure_dyn[wk]["03"],
            }
            for wk, ck in zip(
                wave_adds["03"][:3],
                exp_method_03.multimanagers[summoner_id].afterwards.allCounts.keys(),
            )
        },
    )
    quantity_container = exp_method_03.multimanagers[summoner_id].quantity_container
    for rk, report in quantity_container.items():
        for qk, quantities in report.items():
            for qqi, quantity in enumerate(quantities):
                assert isinstance(quantity, dict), f"The quantity is not a dict: {quantity}."
                assert all(["entropy" in quantity, "purity" in quantity]), (
                    "The necessary quantities 'entropy', 'purity' "
                    + f"are not found: {quantity.keys()}-{qk}-{rk}."
                )
                assert np.abs(quantity["purity"] - answer_list[qqi]) < THREDHOLD, (
                    "The randomized measurement result is wrong: "
                    + f"{np.abs(quantity['purity'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['purity']} != {answer_list[qqi]}."
                )

    read_summoner_id = exp_method_03.multiRead(
        summoner_name=exp_method_03.multimanagers[summoner_id].summoner_name,
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )

    assert (
        read_summoner_id == summoner_id
    ), f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."


def test_export():
    """Export the results."""

    quickJSON(
        results,
        f"test_qurrent.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
