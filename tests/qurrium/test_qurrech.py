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
import numpy as np

from utils import CNOTDynCase4To8, DummyTwoBodyWithDedicatedClbits, current_time_filename

from qurry.qurrech import EchoListen
from qurry.tools.backend import GeneralSimulator
from qurry.tools.backend.import_simulator import SIM_DEFAULT_SOURCE, SIM_IMPORT_ERROR_INFOS
from qurry.capsule import mori, hoshi, quickRead, quickJSON
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet
from qurry.exceptions import QurryDependenciesNotWorking


tag_list = mori.TagList()
statesheet = hoshi.Hoshi()

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "random_unitary_seeds.json")
SEED_SIMULATOR = 2019  # <harmony/>
THREDHOLD = 0.26
MANUAL_ASSERT_ERROR = False

exp_method_01 = EchoListen(method="hadamard")
exp_method_02 = EchoListen(method="randomized")
exp_method_02_with_extra_clbits = EchoListen(method="randomized")
exp_method_03 = EchoListen(method="randomized_v1")

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
    "02_true_overlap": [],
}
answer = {}
measure_dyn = {}

results = {
    "hadamard": {},
    "randomized": {},
    "randomized_with_extra_clbits": {},
    "randomized_true_overlap": {},
    "randomized_v1": {},
}

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
    answer[f"{i}-topological-period"] = 0.5
    seed_usage[f"{i}-topological-period"] = i
    measure_dyn[f"{i}-topological-period"] = {
        "01": (0, 2),
        "02": range(-2, 0),
        "03": (0, 2),
    }
    # purity = 0.25

    if SIM_DEFAULT_SOURCE == "qiskit_aer":
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

        cross_test_name = (
            f"{i}-entangle-by-dyn",
            exp_method_02_with_extra_clbits.add(
                CNOTDynCase4To8(i, export="comparison"), f"{i}-entangle-by-dyn-comparison"
            ),
        )
        wave_adds["02_true_overlap"].append(cross_test_name)
        answer[cross_test_name] = 1.0
        seed_usage[cross_test_name] = i
        measure_dyn[cross_test_name] = {
            "02_true_overlap": [0, i - 1],
        }
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
    """Test the quantity of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    exp_id = exp_method_01.measure(tgt, tgt, (0, 2), backend=backend)
    exp_method_01.exps[exp_id].analyze()
    quantity = exp_method_01.exps[exp_id].reports[0].content._asdict()
    assert all(
        ["echo" in quantity]
    ), f"The necessary quantities 'echo' are not found: {quantity.keys()}."

    diff = np.abs(quantity["echo"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The hadamard test result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity['echo']} != {answer[tgt]}."
    )

    results["hadamard"][tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "target_quantity": quantity["echo"],
        "is_correct": is_correct,
    }


def test_multi_output_01():
    """Test the multi-output of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave1": k,
            "wave2": k,
            "degree": measure_dyn[k]["01"],
        }
        for k in wave_adds["01"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["01"][:3]]

    summoner_id = exp_method_01.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name="qurrech_hadamard",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_01.multiAnalysis(summoner_id)
    quantity_container = exp_method_01.multimanagers[summoner_id].quantity_container
    for rk, report in quantity_container.items():
        for qk, quantities in report.items():
            for qqi, quantity in enumerate(quantities):
                assert isinstance(quantity, dict), f"The quantity is not a dict: {quantity}."
                assert all(
                    ["echo" in quantity]
                ), f"The necessary quantities 'echo' are not found: {quantity.keys()}-{qk}-{rk}."
                assert np.abs(quantity["echo"] - answer_list[qqi]) < THREDHOLD, (
                    "The hadamard test result is wrong: "
                    + f"{np.abs(quantity['echo'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['echo']} != {answer_list[qqi]}."
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
    """Test the quantity of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    # pylint: disable=unexpected-keyword-arg
    exp_id = exp_method_02.measure(
        wave1=tgt,
        wave2=tgt,
        times=20,
        second_transpile_args={
            "optimization_level": 3,
        },
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(20)},
        backend=backend,
    )
    # pylint: enable=unexpected-keyword-arg
    exp_method_02.exps[exp_id].analyze(measure_dyn[tgt]["02"])
    quantity = exp_method_02.exps[exp_id].reports[0].content._asdict()
    assert all(
        ["echo" in quantity]
    ), f"The necessary quantities 'echo' are not found: {quantity.keys()}."

    diff = np.abs(quantity["echo"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity['echo']} != {answer[tgt]}."
    )
    results["randomized"][tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "target_quantity": quantity["echo"],
        "is_correct": is_correct,
    }


def test_multi_output_02():
    """Test the multi-output of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave1": k,
            "wave2": k,
            "times": 20,
            "transpile_args": {
                "optimization_level": 2,
            },
            "random_unitary_seeds": {i: random_unitary_seeds[seed_usage[k]][i] for i in range(20)},
        }
        for k in wave_adds["02"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["02"][:3]]

    summoner_id = exp_method_02.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name="qurrech_randomized",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
    )
    summoner_id = exp_method_02.multiAnalysis(
        summoner_id,
        specific_analysis_args={
            ck: {
                "selected_classical_registers": measure_dyn[wk]["02"],
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
                assert all(
                    ["echo" in quantity]
                ), f"The necessary quantities 'echo' are not found: {quantity.keys()}-{qk}-{rk}."
                assert np.abs(quantity["echo"] - answer_list[qqi]) < THREDHOLD, (
                    "The randomized measurement result is wrong: "
                    + f"{np.abs(quantity['echo'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['echo']} != {answer_list[qqi]}."
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

    if SIM_DEFAULT_SOURCE == "qiskit_aer":
        # pylint: disable=unexpected-keyword-arg
        exp_id = exp_method_02_with_extra_clbits.measure(
            wave1=tgt,
            wave2=tgt,
            times=50,
            measure_1=measure_dyn[tgt]["02_with_extra_clbits"],
            measure_2=measure_dyn[tgt]["02_with_extra_clbits"],
            random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(50)},
            backend=backend,
        )
        # pylint: enable=unexpected-keyword-arg
        exp_method_02_with_extra_clbits.exps[exp_id].write(
            save_location=os.path.join(os.path.dirname(__file__), "exports")
        )
        analysis = exp_method_02_with_extra_clbits.exps[exp_id].analyze(
            measure_dyn[tgt]["02_with_extra_clbits"]
        )
        quantity = analysis.content._asdict()
        exp_method_02_with_extra_clbits.exps[exp_id].write(
            save_location=os.path.join(os.path.dirname(__file__), "exports")
        )

        assert all(
            ["echo" in quantity]
        ), f"The necessary quantities 'echo' are not found: {quantity.keys()}."

        diff = np.abs(quantity["echo"] - answer[tgt])
        is_correct = diff < THREDHOLD
        assert (not MANUAL_ASSERT_ERROR) and is_correct, (
            "The randomized measurement result is wrong: "
            + f"{diff} !< {THREDHOLD}."
            + f" {quantity['echo']} != {answer[tgt]}. exp_id: {exp_id}."
        )

        results["randomized_with_extra_clbits"][tgt] = {
            "answer": answer[tgt],
            "difference": diff,
            "target_quantity": quantity["echo"],
            "is_correct": is_correct,
        }
    else:
        warnings.warn(
            f'The backend is {SIM_DEFAULT_SOURCE} instead of "qiskit_aer" '
            + "which is guaranteed to work with dynamic circuit. "
            + f"And here is the error message: {SIM_IMPORT_ERROR_INFOS['qiskit_aer']}.",
            category=QurryDependenciesNotWorking,
        )


@pytest.mark.parametrize("tgt", wave_adds["02_true_overlap"])
def test_quantity_02_true_overlap(tgt):
    """Test the quantity of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    if SIM_DEFAULT_SOURCE == "qiskit_aer":
        # pylint: disable=unexpected-keyword-arg
        exp_id = exp_method_02_with_extra_clbits.measure(
            wave1=tgt[0],
            wave2=tgt[1],
            times=20,
            measure_1=measure_dyn[tgt]["02_true_overlap"],
            measure_2=measure_dyn[tgt]["02_true_overlap"],
            random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(20)},
            backend=backend,
        )
        # pylint: enable=unexpected-keyword-arg
        exp_method_02_with_extra_clbits.exps[exp_id].analyze(measure_dyn[tgt]["02_true_overlap"])
        quantity = exp_method_02_with_extra_clbits.exps[exp_id].reports[0].content._asdict()
        assert all(
            ["echo" in quantity]
        ), f"The necessary quantities 'echo' are not found: {quantity.keys()}."

        diff = np.abs(quantity["echo"] - answer[tgt])
        is_correct = diff < THREDHOLD
        assert (not MANUAL_ASSERT_ERROR) and is_correct, (
            "The randomized measurement result is wrong: "
            + f"{diff} !< {THREDHOLD}."
            + f" {quantity['echo']} != {answer[tgt]}."
        )
        results["randomized_true_overlap"][tgt] = {
            "answer": answer[tgt],
            "difference": diff,
            "target_quantity": quantity["echo"],
            "is_correct": is_correct,
        }
    else:
        warnings.warn(
            f'The backend is {SIM_DEFAULT_SOURCE} instead of "qiskit_aer" '
            + "which is guaranteed to work with dynamic circuit. "
            + f"And here is the error message: {SIM_IMPORT_ERROR_INFOS['qiskit_aer']}.",
            category=QurryDependenciesNotWorking,
        )


@pytest.mark.parametrize("tgt", wave_adds["03"])
def test_quantity_03(tgt):
    """Test the quantity of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    # pylint: disable=unexpected-keyword-arg
    exp_id = exp_method_03.measure(
        wave1=tgt,
        wave2=tgt,
        times=20,
        random_unitary_seeds={i: random_unitary_seeds[seed_usage[tgt]][i] for i in range(20)},
        backend=backend,
    )
    # pylint: enable=unexpected-keyword-arg
    exp_method_03.exps[exp_id].analyze(measure_dyn[tgt]["03"])
    quantity = exp_method_03.exps[exp_id].reports[0].content._asdict()
    assert all(
        ["echo" in quantity]
    ), f"The necessary quantities 'echo' are not found: {quantity.keys()}."

    diff = np.abs(quantity["echo"] - answer[tgt])
    is_correct = diff < THREDHOLD
    assert (not MANUAL_ASSERT_ERROR) and is_correct, (
        "The randomized measurement result is wrong: "
        + f"{diff} !< {THREDHOLD}."
        + f" {quantity['echo']} != {answer[tgt]}."
    )

    results["randomized_v1"][tgt] = {
        "answer": answer[tgt],
        "difference": diff,
        "target_quantity": quantity["echo"],
        "is_correct": is_correct,
    }


def test_multi_output_03():
    """Test the multi-output of echo.

    Args:
        tgt (Hashable): The target wave key in Qurry.
    """

    config_list = [
        {
            "wave1": k,
            "wave2": k,
            "times": 20,
            "random_unitary_seeds": {i: random_unitary_seeds[seed_usage[k]][i] for i in range(20)},
        }
        for k in wave_adds["03"][:3]
    ]
    answer_list = [answer[k] for k in wave_adds["03"][:3]]

    summoner_id = exp_method_03.multiOutput(
        config_list,  # type: ignore
        backend=backend,
        summoner_name="qurrech_randomized",
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
                assert all(
                    ["echo" in quantity]
                ), f"The necessary quantities 'echo' are not found: {quantity.keys()}-{qk}-{rk}."
                assert np.abs(quantity["echo"] - answer_list[qqi]) < THREDHOLD, (
                    "The randomized measurement result is wrong: "
                    + f"{np.abs(quantity['echo'] - answer_list[qqi])} !< {THREDHOLD}."
                    + f" {quantity['echo']} != {answer_list[qqi]}."
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
        f"test_qurrech.{current_time_filename()}.json",
        mode="w",
        save_location=os.path.join(os.path.dirname(__file__), "exports"),
        jsonable=True,
    )
