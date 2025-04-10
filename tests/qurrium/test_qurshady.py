"""
================================================================
Test the qurry.qurrent module ShadowUnveil class
================================================================

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
import warnings
import pytest
import numpy as np

from utils import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits, current_time_filename

from qurry.qurrent import ShadowUnveil
from qurry.tools.backend import GeneralSimulator
from qurry.tools.backend.import_simulator import SIM_DEFAULT_SOURCE, SIM_IMPORT_ERROR_INFOS
from qurry.capsule import mori, hoshi, quickRead, quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet
from qurry.exceptions import QurryDependenciesNotWorking


tag_list = mori.TagList()
statesheet = hoshi.Hoshi()

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")
SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.25
MANUAL_ASSERT_ERROR = False

exp_method_01 = ShadowUnveil()
exp_method_01_with_extra_clbits = ShadowUnveil()

random_unitary_seeds_raw: dict[str, dict[str, dict[str, int]]] = quickRead(FILE_LOCATION)
random_unitary_seeds = {
    int(k): {int(k2): {int(k3): v3 for k3, v3 in v2.items()} for k2, v2 in v.items()}
    for k, v in random_unitary_seeds_raw.items()
}
seed_usage = {}

wave_adds = {
    "01": [],
    "01_with_extra_clbits": [],
}
answer = {}
measure_dyn = {}

results = {
    "classical_shadow": {},
    "classical_shadow_with_extra_clbits": {},
}

for i in range(4, 7, 2):
    wave_adds["01"].append(exp_method_01.add(TrivialParamagnet(i), f"{i}-trivial"))
    answer[f"{i}-trivial"] = 1.0
    seed_usage[f"{i}-trivial"] = i
    measure_dyn[f"{i}-trivial"] = {
        "01": range(-2, 0),
    }
    # purity = 1.0

    wave_adds["01"].append(exp_method_01.add(GHZ(i), f"{i}-GHZ"))
    answer[f"{i}-GHZ"] = 0.5
    seed_usage[f"{i}-GHZ"] = i
    measure_dyn[f"{i}-GHZ"] = {
        "01": range(-2, 0),
    }
    # purity = 0.5

    wave_adds["01"].append(
        exp_method_01.add(TopologicalParamagnet(i, "period"), f"{i}-topological-period")
    )
    answer[f"{i}-topological-period"] = 0.25
    seed_usage[f"{i}-topological-period"] = i
    measure_dyn[f"{i}-topological-period"] = {
        "01": range(-2, 0),
    }
    # purity = 0.25

    if SIM_DEFAULT_SOURCE == "qiskit_aer":
        wave_adds["01_with_extra_clbits"].append(
            exp_method_01_with_extra_clbits.add(CNOTDynCase4To8(i), f"{i}-entangle-by-dyn")
        )
        answer[f"{i}-entangle-by-dyn"] = 1
        seed_usage[f"{i}-entangle-by-dyn"] = i
        measure_dyn[f"{i}-entangle-by-dyn"] = {
            "01_with_extra_clbits": [0, i - 1],
        }
        # purity = 1, de-facto all system when selected qubits is [0, i - 1]

        wave_adds["01_with_extra_clbits"].append(
            exp_method_01_with_extra_clbits.add(CNOTDynCase4To8(i), f"{i}-entangle-by-dyn-half")
        )
        answer[f"{i}-entangle-by-dyn-half"] = 0.5
        seed_usage[f"{i}-entangle-by-dyn-half"] = i
        measure_dyn[f"{i}-entangle-by-dyn-half"] = {
            "01_with_extra_clbits": [0],
        }
        # purity = 0.5, when selected qubits is [0]

        wave_adds["01_with_extra_clbits"].append(
            exp_method_01_with_extra_clbits.add(
                DummyTwoBodyWithDedicatedClbits(i), f"{i}-dummy-2-body-with-clbits"
            )
        )
        answer[f"{i}-dummy-2-body-with-clbits"] = 1.0
        seed_usage[f"{i}-dummy-2-body-with-clbits"] = i
        measure_dyn[f"{i}-dummy-2-body-with-clbits"] = {
            "01_with_extra_clbits": [i - 2, i - 1],
        }
        # purity = 1.0
    else:
        warnings.warn(
            f'The backend is {SIM_DEFAULT_SOURCE} instead of "qiskit_aer" '
            + "which is guaranteed to work with dynamic circuit. "
            + f"And here is the error message: {SIM_IMPORT_ERROR_INFOS['qiskit_aer']}.",
            category=QurryDependenciesNotWorking,
        )

backend = GeneralSimulator()
# backend = BasicAer.backends()[0]
backend.set_options(seed_simulator=SEED_SIMULATOR)  # type: ignore


@pytest.mark.parametrize("tgt", wave_adds["01"])
def test_quantity_01(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_01.measure(
        wave=tgt,
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(100)},
        backend=backend,
    )
    analysis_01 = exp_method_01.exps[exp_id].analyze(measure_dyn[tgt]["01"])
    quantity_01 = analysis_01.content._asdict()
    # analysis_02 = exp_method_01.exps[exp_id].analyze(measure_dyn[tgt]["01"], counts_used=range(5))
    # quantity_02 = analysis_02.content._asdict()
    # analysis_03 = exp_method_01.exps[exp_id].analyze(measure_dyn[tgt]["01"], counts_used=range(5))
    # quantity_03 = analysis_03.content._asdict()

    assert all(
        ["entropy" in quantity_01, "purity" in quantity_01]
    ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
    # TODO: Error mitigation will be added in the future.
    # assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
    #     "The all system entropy is not changed: "
    #     + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
    #     + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
    # )
    # assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
    #     "The all system entropy is not changed: "
    #     + f"{quantity_03['entropyAllSys']} != {quantity_02['entropyAllSys']}."
    # )
    # assert (
    #     quantity_02["all_system_source"] == "independent"
    # ), f"The source of all system is not independent: {quantity_02['all_system_source']}."
    # assert (
    #     "AnalysisHeader" in quantity_03["all_system_source"]
    # ), f"The source of all system is not from existed analysis:
    # {quantity_03['all_system_source']}."

    diff = np.abs(quantity_01["purity"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity_01['purity']} != {answer[tgt]}."
    )
    results["classical_shadow"][tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "target_quantity": quantity_01["purity"],
        "is_correct": is_correct,
    }


def test_multi_output_01():
    """Test the multi-output of purity and entropy.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave": k,
            "times": 50,
            "random_unitary_seeds": {i: random_unitary_seeds[seed_usage[k]][i] for i in range(50)},
        }
        for k in wave_adds["01"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["01"][:3]]

    summoner_id = exp_method_01.multiOutput(
        config_list,  # type: ignore
        shots=2048,
        backend=backend,
        summoner_name="qurshady",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_01.multiAnalysis(
        summoner_id,
        specific_analysis_args={
            ck: {
                "selected_qubits": measure_dyn[wk]["01"],
            }
            for wk, ck in zip(
                wave_adds["01"][:3],
                exp_method_01.multimanagers[summoner_id].afterwards.allCounts.keys(),
            )
        },
    )
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
                    "The randomized measurement result is wrong: "
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


@pytest.mark.parametrize("tgt", wave_adds["01_with_extra_clbits"])
def test_quantity_01_with_extra_clbits(tgt):
    """Test the quantity of entropy and purity.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    if SIM_DEFAULT_SOURCE == "qiskit_aer":
        exp_id = exp_method_01_with_extra_clbits.measure(
            wave=tgt,
            times=100,
            measure=measure_dyn[tgt]["01_with_extra_clbits"],
            random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(100)},
            backend=backend,
        )
        analysis_01 = exp_method_01_with_extra_clbits.exps[exp_id].analyze(
            measure_dyn[tgt]["01_with_extra_clbits"]
        )
        quantity_01 = analysis_01.content._asdict()
        # analysis_02 = exp_method_01_with_extra_clbits.exps[exp_id].analyze(
        #     measure_dyn[tgt]["01_with_extra_clbits"], counts_used=range(5)
        # )
        # quantity_02 = analysis_02.content._asdict()
        # analysis_03 = exp_method_01_with_extra_clbits.exps[exp_id].analyze(
        #     measure_dyn[tgt]["01_with_extra_clbits"], counts_used=range(5)
        # )
        # quantity_03 = analysis_03.content._asdict()

        assert all(
            ["entropy" in quantity_01, "purity" in quantity_01]
        ), f"The necessary quantities 'entropy', 'purity' are not found: {quantity_01.keys()}."
        # TODO: Error mitigation will be added in the future.
        # assert quantity_02["entropyAllSys"] != quantity_01["entropyAllSys"], (
        #     "The all system entropy is not changed: "
        #     + f"counts_used: {quantity_01['counts_used']}: {quantity_02['entropyAllSys']}, "
        #     + f"counts_used: {quantity_02['counts_used']}: {quantity_02['entropyAllSys']},"
        # )
        # assert np.abs(quantity_03["entropyAllSys"] - quantity_02["entropyAllSys"]) < 1e-12, (
        #     "The all system entropy is not changed: "
        #     + f"{quantity_03['entropyAllSys']} != {quantity_02['entropyAllSys']}."
        # )
        # assert (
        #     quantity_02["all_system_source"] == "independent"
        # ), f"The source of all system is not independent: {quantity_02['all_system_source']}."
        # assert "AnalysisHeader" in quantity_03["all_system_source"], (
        #     "The source of all system is not "
        #     + f"from existed analysis: {quantity_03['all_system_source']}."
        # )

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
    else:
        warnings.warn(
            f'The backend is {SIM_DEFAULT_SOURCE} instead of "qiskit_aer" '
            + "which is guaranteed to work with dynamic circuit. "
            + f"And here is the error message: {SIM_IMPORT_ERROR_INFOS['qiskit_aer']}.",
            category=QurryDependenciesNotWorking,
        )


def test_export():
    """Export the results."""

    quickJSON(
        results,
        f"test_qurshady.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
