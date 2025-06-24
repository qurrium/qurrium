"""Experiment Utilities (:mod:`qurry.qurrium.experiment.utils`)"""

import os
import warnings
from uuid import uuid4, UUID
from typing import Optional, Union
from collections.abc import Hashable
from pathlib import Path
import numpy as np

from qiskit import QuantumCircuit

from .arguments import Commonparams, ArgumentsPrototype
from .beforewards import Before
from .afterwards import After
from .analyses import AnalysesContainer
from ..utils.iocontrol import RJUST_LEN
from ...capsule.hoshi import Hoshi
from ...exceptions import (
    QurryHashIDInvalid,
    QurrySummonerInvalid,
    QurryInvalidInherition,
    UnconfiguredWarning,
)


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
            category=UnconfiguredWarning,
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

    Raises:
        QurrySummonerInvalid: If the summoner information is not completed.

    Returns:
        bool: True if the summoner information is completed, False otherwise.
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
        summon_msg.print()
        raise QurrySummonerInvalid(
            "Summoner data is not completed, it will export in single experiment mode.",
        )
    return summon_fulfill


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


def create_save_location(
    save_location: Optional[Union[str, Path]],
    commons: Optional[Commonparams] = None,
) -> Path:
    """Create a save location for the experiment.

    Args:
        save_location (Optional[str]): The save location of the experiment.
        commons (Optional[Commonparams]):
            The common parameters of the experiment.
            It is used to get the default save location if `save_location` is None.

    Returns:
        Path: The save location as a Path object.

    Raises:
        ValueError:
            If `save_location` is not a Path or str,
            or if it is None and `commons` is also None.
    """
    if isinstance(save_location, Path):
        return save_location
    if isinstance(save_location, str):
        return Path(save_location)
    if save_location is None and commons is not None:
        if commons.save_location is None:
            raise ValueError("save_location is None, please provide a valid save_location")
        return Path(commons.save_location)

    raise ValueError(f"save_location must be Path or str, not {type(save_location)}")


def folder_with_repeat_times(exp_name: str, repeat_times: int) -> str:
    """Create a folder with repeat times.

    Args:
        exp_name (str): The name of the experiment.
        repeat_times (int, optional): The repeat times of the experiment. Defaults to 1.

    Returns:
        str: The folder name with repeat times.
    """
    return f"./{exp_name}.{str(repeat_times).rjust(RJUST_LEN, '0')}/"


def decide_folder_and_filename(commons: Commonparams, args: ArgumentsPrototype) -> tuple[str, str]:
    """Decide the folder and filename for the experiment.

    Args:
        commons (Commonparams): The common parameters of the experiment.
        args (ArgumentsPrototype): The arguments of the experiment.

    Returns:
        tuple[str, str]: The folder and filename for the experiment.
    """

    if all(v is not None for v in [commons.serial, commons.summoner_id, commons.summoner_id]):
        folder = f"./{commons.summoner_name}/"
        filename = f"index={commons.serial}.id={commons.exp_id}"
        return folder, filename

    repeat_times = 1
    folder = folder_with_repeat_times(args.exp_name, repeat_times)
    while os.path.exists(folder):
        repeat_times += 1
        folder = folder_with_repeat_times(args.exp_name, repeat_times)
    filename = f"{args.exp_name}.{str(repeat_times).rjust(RJUST_LEN, '0')}.id={commons.exp_id}"
    return folder, filename
