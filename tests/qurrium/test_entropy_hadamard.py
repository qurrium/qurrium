"""Test the Qurrium Runtime :class:`EntropyMeasureHadamard`.

It's from :class:`~qurry.qurry.qurries.entropy_hadamard.qurry.EntropyMeasureHadamard`.

- hadamard test at shots = 1024
    - [4-trivial] 0.0 <= 0.25. 1.0 ~= 1.0
    - [4-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5
    - [4-topological-period] 0.033203125 <= 0.25. 0.283203125 ~= 0.25
    - [6-trivial] 0.0 <= 0.25. 1.0 ~= 1.0
    - [6-GHZ] 0.005859375 <= 0.25. 0.505859375 ~= 0.5
    - [6-topological-period] 0.041015625 <= 0.25. 0.291015625 ~= 0.25

"""

import logging
import pytest

from qurry.qurries.entropy_hadamard import EntropyMeasureHadamard, EMHMeasureArgs
from qurry.qurries.entropy_hadamard.analysis import EMHAnalyzeArgs, EMHAnalysis
from qurry.recipe import TrivialParamagnet, GHZ, TopologicalParamagnet

from utilities.simulator import get_seeded_simulator
from utilities.other import (
    CaseEntriesTuple,
    CaseDataDict,
    check_analysis_result,
    EXPORT_DIR,
    make_specific_analysis_args,
)

logger = logging.getLogger(__name__)

SIMULATOR = get_seeded_simulator()

case_datas: list[CaseDataDict] = [
    {"circuit": TrivialParamagnet(4, name="4-trivial"), "expect_answer": 1.0},
    {"circuit": GHZ(4, name="4-GHZ"), "expect_answer": 0.5},
    {"circuit": TopologicalParamagnet(4, name="4-topological-period"), "expect_answer": 0.25},
    {"circuit": TrivialParamagnet(6, name="6-trivial"), "expect_answer": 1.0},
    {"circuit": GHZ(6, name="6-GHZ"), "expect_answer": 0.5},
    {"circuit": TopologicalParamagnet(6, name="6-topological-period"), "expect_answer": 0.25},
]

CASES: list[CaseEntriesTuple[EMHMeasureArgs, EMHAnalyzeArgs]] = [
    CaseEntriesTuple(
        tags=("hadamard", case_data["circuit"].name),
        measure_entries={
            "wave": case_data["circuit"],
            "degree": (0, 2),
            "backend": SIMULATOR,
            "shots": 1024,
        },
        analyze_entries={},
        expect_answer=case_data["expect_answer"],
    )
    for case_data in case_datas
]

THREDHOLD = 0.25


@pytest.mark.parametrize("case_entries", CASES)
def test_measure_and_analyze(
    case_entries: CaseEntriesTuple[EMHMeasureArgs, EMHAnalyzeArgs],
) -> None:
    """Test orphan experiments.

    Args:
        case_entries (CaseEntriesTuple[EMHMeasureArgs, EMHAnalyzeArgs]): The test case item.
    """

    exp_method = EntropyMeasureHadamard()
    exp_id = exp_method.measure(**case_entries.measure_entries_with_tags())
    analysis_01 = exp_method.exps[exp_id].analyze(**case_entries.analyze_entries)

    check_report = check_analysis_result(
        analysis_01.results["default"],
        target_field="purity",
        expect_answer=case_entries.expect_answer,
        name=case_entries.name,
        threshold=THREDHOLD,
    )

    logger.info(check_report.make_logger())


def test_multi_output_all() -> None:
    """Test the multi-output experiment for all cases."""

    exp_method = EntropyMeasureHadamard()

    config_list = []
    cases_with_tags: dict[tuple[str, ...], CaseEntriesTuple[EMHMeasureArgs, EMHAnalyzeArgs]] = {}
    for i, case_entries in enumerate(CASES):
        config = case_entries.measure_entries_with_tags(f"index_{i}")
        if "tags" not in config:
            config["tags"] = (f"index_{i}",)
        config_list.append(config)
        cases_with_tags[config["tags"]] = case_entries  # type: ignore

    summoner_id = exp_method.multiOutput(
        config_list,
        backend=SIMULATOR,
        summoner_name="entropy_hadamard",
        save_location=EXPORT_DIR,
        multiprocess_build=True,
        multiprocess_write=False,
    )

    summoner_id, report_name = exp_method.multiAnalysis(
        summoner_id,
        analysis_name="test_report",
        no_serialize=True,
        specific_analysis_args=make_specific_analysis_args(
            exp_method, summoner_id, cases_with_tags
        ),
    )
    test_report: dict[tuple[str, ...], list[EMHAnalysis]] = exp_method.multimanagers[
        summoner_id
    ].all_reports(report_name)

    for tags, report_list in test_report.items():
        for report in report_list:
            check_report = check_analysis_result(
                report.results["default"],
                target_field="purity",
                expect_answer=cases_with_tags[tags].expect_answer,
                name=cases_with_tags[tags].name,
                threshold=THREDHOLD,
            )
            logger.info(check_report.make_logger() + "| multi-output all")

    read_summoner_id = exp_method.multiRead(
        summoner_name=exp_method.multimanagers[summoner_id].summoner_name,
        save_location=EXPORT_DIR,
    )
    assert read_summoner_id == summoner_id, (
        f"The read summoner id is wrong: {read_summoner_id} != {summoner_id}."
    )
