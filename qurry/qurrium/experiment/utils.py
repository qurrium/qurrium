"""Experiment Utilities (:mod:`qurry.qurrium.experiment.utils`)"""

import warnings
from uuid import uuid4, UUID
from typing import Optional, Union
from collections.abc import Hashable
import numpy as np

from qiskit import QuantumCircuit

from .arguments import Commonparams, ArgumentsPrototype
from .beforewards import Before
from .afterwards import After
from .analyses import AnalysesContainer
from ...capsule.hoshi import Hoshi
from ...exceptions import (
    QurryHashIDInvalid,
    QurryInvalidInherition,
    QurryWarning,
    QurrySummonerInfoIncompletion,
)


EXPERIMENT_UNEXPORTS = ["side_product", "result", "circuits"]
"""Unexports properties."""
DEPRECATED_PROPERTIES = ["figTranspiled", "fig_original"]
"""Deprecated properties.
    - `figTranspiled` is deprecated since v0.6.0.
    - `fig_original` is deprecated since v0.6.10.
"""


def exp_id_process(exp_id: Optional[str]) -> str:
    """Check the exp_id is valid or not, if not, then generate a new one.

    Args:
        exp_id (Optional[str]): The id of the experiment to be checked.

    Returns:
        str: The valid exp_id.
    """

    if exp_id is None:
        return str(uuid4())

    try:
        UUID(exp_id, version=4)
    except ValueError as e:
        exp_id = None
        warnings.warn(
            f"exp_id is not a valid UUID, it will be generated automatically.\n{e}",
            category=QurryHashIDInvalid,
        )
    else:
        return exp_id
    return str(uuid4())


def memory_usage_factor_expect(
    target: list[tuple[Hashable, Union[QuantumCircuit, str]]],
    circuits: list[QuantumCircuit],
    commonparams: Commonparams,
) -> int:
    """Estimate the memory usage of :cls:`ExperimentPrototype` by the circuits.

    The memory usage is estimated by the number of instructions in the circuits and
    the number of shots. The factor is calculated by the formula:

    .. code-block:: txt
        factor = target_circuit_instructions_num + sqrt(shots) * target_circuit_instructions_num

    where `target_circuit_instructions_num` is the number of instructions in the target circuits,
    `transpiled_circuit_instructions_num` is the number of instructions in the circuits
    which has been transpiled and will be run on the backend,
    and `shots` is the number of shots.

    The factor is rounded to the nearest integer.
    The factor is used to estimate the memory usage of the experiment.

    Args:
        circuits (list[QuantumCircuit]): The circuits to be estimated.
        commonparams (Commonparams): The common parameters of the experiment.

    Returns:
        int: The factor of the memory usage.
    """

    circuit_instructions_num = sum(len(circuit.data) for circuit in circuits)

    factor = circuit_instructions_num * np.sqrt(commonparams.shots)
    factor += sum(len(circuit.data) for _, circuit in target if isinstance(circuit, QuantumCircuit))

    return int(np.round(factor))


def implementation_check(
    name_exps: str,
    args: ArgumentsPrototype,
    commons: Commonparams,
) -> None:
    """Check whether the experiment is implemented correctly."""
    duplicate_fields = set(args._fields) & set(commons._fields)
    if len(duplicate_fields) > 0:
        raise QurryInvalidInherition(
            f"{name_exps}.arguments which and {name_exps}.commonparams "
            f"should not have same fields: {duplicate_fields}."
        )
    if name_exps == "ExperimentPrototype":
        warnings.warn(
            "You should set a new __name__ for your experiment class, "
            + "otherwise it will be considered as an abstract class of Qurrium during printing.",
            category=QurryWarning,
        )


def summonner_check(
    serial: Optional[int],
    summoner_id: Optional[str],
    summoner_name: Optional[str],
):
    """Check the summoner information taken from the experiment.

    Args:
        serial (Optional[int]): The serial number of the experiment.
        summoner_id (Optional[str]): The ID of the summoner.
        summoner_name (Optional[str]): The name of the summoner.
    """

    summon_check = {
        "serial": serial,
        "summoner_id": summoner_id,
        "summoner_name": summoner_name,
    }
    summon_detect = any((v is not None) for v in summon_check.values())
    summon_fulfill = all((v is not None) for v in summon_check.values())
    if summon_detect and not summon_fulfill:
        summon_msg = Hoshi(ljust_description_len=20)
        summon_msg.newline(("divider",))
        summon_msg.newline(("h3", "Summoner Info Incompletion"))
        summon_msg.newline(("itemize", "Summoner info detect.", summon_detect))
        summon_msg.newline(("itemize", "Summoner info fulfilled.", summon_fulfill))
        for k, v in summon_check.items():
            summon_msg.newline(("itemize", k, str(v), f"fulfilled: {v is not None}", 2))
        warnings.warn(
            "Summoner data is not completed, it will export in single experiment mode.",
            category=QurrySummonerInfoIncompletion,
        )
        summon_msg.print()


def make_statesheet(
    exp_name: str,
    args: ArgumentsPrototype,
    commons: Commonparams,
    outfields: dict[str, str],
    beforewards: Before,
    afterwards: After,
    reports: AnalysesContainer,
    report_expanded: bool = False,
    hoshi: bool = False,
) -> Hoshi:
    """Show the state of experiment.

    Args:
        exp_name (str): Name of the experiment.
        args (ArgumentsPrototype): Arguments of the experiment.
        commons (Commonparams): Common parameters of the experiment.
        outfields (dict[str, str]): Unused arguments.
        beforewards (Before): Beforewards of the experiment.
        afterwards (After): Afterwards of the experiment.
        reports (AnalysesContainer): Reports of the experiment.
        report_expanded (bool, optional): Show more infomation. Defaults to False.
        hoshi (bool, optional): Showing name of Hoshi. Defaults to False.

    Returns:
        Hoshi: Statesheet of experiment.
    """

    info = Hoshi(
        [
            ("h1", f"{exp_name} with exp_id={commons.exp_id}"),
        ],
        name="Hoshi" if hoshi else "QurryExperimentSheet",
    )
    info.newline(("itemize", "arguments"))
    for k, v in args._asdict().items():
        info.newline(("itemize", str(k), str(v), "", 2))

    info.newline(("itemize", "commonparams"))
    for k, v in commons._asdict().items():
        info.newline(
            (
                "itemize",
                str(k),
                str(v),
                (),
                2,
            )
        )

    info.newline(
        (
            "itemize",
            "outfields",
            len(outfields),
            "Number of unused arguments.",
            1,
        )
    )
    for k, v in outfields.items():
        info.newline(("itemize", str(k), v, "", 2))

    info.newline(("itemize", "beforewards"))
    for k, v in beforewards._asdict().items():
        if isinstance(v, str):
            info.newline(("itemize", str(k), str(v), "", 2))
        else:
            info.newline(("itemize", str(k), len(v), f"Number of {k}", 2))

    info.newline(("itemize", "afterwards"))
    for k, v in afterwards._asdict().items():
        if k == "job_id":
            info.newline(
                (
                    "itemize",
                    str(k),
                    str(v),
                    "If it's null meaning this experiment "
                    + "doesn't use online backend like IBMQ.",
                    2,
                )
            )
        elif isinstance(v, str):
            info.newline(("itemize", str(k), str(v), "", 2))
        else:
            info.newline(("itemize", str(k), len(v), f"Number of {k}", 2))

    info.newline(("itemize", "reports", len(reports), "Number of analysis.", 1))
    if report_expanded:
        for ser, item in reports.items():
            info.newline(
                (
                    "itemize",
                    "serial",
                    f"k={ser}, serial={item.header.serial}",
                    None,
                    2,
                )
            )
            info.newline(("txt", item, 3))

    return info
