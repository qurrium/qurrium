"""Multi-process component for multimanager (:mod:`qurry.qurrium.multimanager.process`)"""

from typing import Any
from pathlib import Path

from .arguments import MultiCommonparams
from ..container import _E
from ..experiment import ExperimentPrototype, Export
from ..utils.iocontrol import IOComplex


def multiprocess_exporter(
    id_exec: str,
    exps_export: Export,
    mode: str = "w+",
    indent: int = 2,
    encoding: str = "utf-8",
    jsonable: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Multiprocess exporter and writer for experiment.

    Args:
        id_exec (Hashable): ID of experiment.
        exps_export (Export): The export of experiment.
        mode (str, optional): The mode of writing. Defaults to "w+".
        indent (int, optional): The indent of writing. Defaults to 2.
        encoding (str, optional): The encoding of writing. Defaults to "utf-8".
        jsonable (bool, optional): The jsonable of writing. Defaults to False.

    Returns:
        tuple[Hashable, dict[str, Any]]: The ID of experiment and the files of experiment.
    """
    qurryinfo_exp_id, qurryinfo_files = exps_export.write(
        mode=mode,
        indent=indent,
        encoding=encoding,
        jsonable=jsonable,
        mute=True,
        multiprocess=False,
        pbar=None,
    )
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )
    del exps_export

    return qurryinfo_exp_id, qurryinfo_files


def multiprocess_exporter_wrapper(
    all_arguments: tuple[str, Export, str, int, str, bool],
) -> tuple[str, dict[str, str]]:
    """Multiprocess wrapper for exporter.

    Args:
        all_arguments (tuple[str, Export, str, int, str, bool]):
            The arguments for exporter.
            - id_exec (str): ID of experiment.
            - exps_export (Export): The export of experiment.
            - mode (str): The mode of writing.
            - indent (int): The indent of writing.
            - encoding (str): The encoding of writing.
            - jsonable (bool): The jsonable of writing.

    Returns:
        tuple[str, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_exporter(*all_arguments)


def multiprocess_writer(
    id_exec: str,
    exps: ExperimentPrototype,
    save_location: Path,
    export_transpiled_circuit: bool = False,
    mode: str = "w+",
    indent: int = 2,
    encoding: str = "utf-8",
    jsonable: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Multiprocess exporter and writer for experiment.

    Args:
        id_exec (Hashable): ID of experiment.
        exps (ExperimentPrototype): The export of experiment.
        mode (str, optional): The mode of writing. Defaults to "w+".
        save_location (Path): The location of saving.
        export_transpiled_circuit (bool, optional):
            Whether to export transpiled circuit. Defaults to False.
        indent (int, optional): The indent of writing. Defaults to 2.
        encoding (str, optional): The encoding of writing. Defaults to "utf-8".
        jsonable (bool, optional): The jsonable of writing. Defaults to False.

    Returns:
        tuple[Hashable, dict[str, Any]]: The ID of experiment and the files of experiment.
    """
    export_instance = exps.export(
        save_location=save_location,
        export_transpiled_circuit=export_transpiled_circuit,
    )
    qurryinfo_exp_id, qurryinfo_files = export_instance.write(
        mode=mode,
        indent=indent,
        encoding=encoding,
        jsonable=jsonable,
        mute=True,
        multiprocess=False,
        pbar=None,
    )
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )
    del export_instance

    return qurryinfo_exp_id, qurryinfo_files


def multiprocess_writer_wrapper(
    all_arguments: tuple[str, _E, Path, bool, str, int, str, bool],
) -> tuple[str, dict[str, str]]:
    """Multiprocess wrapper for exporter.

    Args:
        all_arguments (tuple[str, ExperimentPrototype, Path, bool, str, int, str, bool, bool]):
            The arguments for exporter.
            - id_exec (str): ID of experiment.
            - exps (ExperimentPrototype): The export of experiment.
            - save_location (Path): The location of saving.
            - export_transpiled_circuit (bool): Whether to export transpiled circuit.
            - mode (str): The mode of writing.
            - indent (int): The indent of writing.
            - encoding (str): The encoding of writing.
            - jsonable (bool): The jsonable of writing.

    Returns:
        tuple[str, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_writer(*all_arguments)


def datetimedict_process(
    multicommons: MultiCommonparams,
    naming_complex: IOComplex,
    multiconfig_name_v5: Path,
    multiconfig_name_v7: Path,
    is_read_or_retrieve: bool,
    read_from_tarfile: bool,
    old_files: dict[str, Any],
):
    """Process the datetime dict of multimanager.

    Args:
        multicommons (MultiCommonparams): The common parameters of multimanager.
        naming_complex (IOComplex): The complex of IO.
        multiconfig_name_v5 (Path): The path of multiConfig in v5.
        multiconfig_name_v7 (Path): The path of multiConfig in v7.
        is_read_or_retrieve (bool): Whether read or retrieve.
        read_from_tarfile (bool): Whether read from tarfile.
        old_files (dict[str, Any]): The old files.
    """

    if "build" not in multicommons.datetimes and not is_read_or_retrieve:
        multicommons.datetimes.add_only("build")

    if naming_complex.tarLocation.exists():
        if (not multiconfig_name_v5.exists()) and (not multiconfig_name_v7.exists()):
            multicommons.datetimes.add_serial("decompress")
        elif read_from_tarfile:
            multicommons.datetimes.add_serial("decompressOverwrite")

    # readV5 files re-export
    if multiconfig_name_v5.exists():
        multicommons.datetimes.add_only("readV7")
        for k in old_files.keys():
            multicommons.files.pop(k, None)
