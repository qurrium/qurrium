"""Multi-process component for multimanager (:mod:`qurry.qurrium.multimanager.process`)"""

from typing import Optional, Any, Type
from pathlib import Path
import gc
import tqdm
import numpy as np

from .arguments import MultiCommonparams
from ..container import _E
from ..experiment.export import Export
from ..utils.iocontrol import IOComplex
from ...tools.parallelmanager import DEFAULT_POOL_SIZE


def multiprocess_builder(
    experiment_instance: Type[_E],
    config: dict[str, Any],
) -> tuple[_E, dict[str, Any]]:
    """Multiprocess builder for experiment.

    Args:
        experiment_instance (Type[_E]): The instance of experiment.
        config (dict[str, Any]): The configuration of experiment.

    Returns:
        tuple[_E, dict[str, Any]]: The instance of experiment and the configuration.
    """
    exp_instance = experiment_instance.build(**config, multiprocess=False)
    return exp_instance, config


def multiprocess_builder_wrapper(
    all_arguments: tuple[Type[_E], dict[str, Any]],
) -> tuple[_E, dict[str, Any]]:
    """Multiprocess builder for exporter.

    Args:
        all_arguments (tuple[Type[_E], dict[str, Any]]):
            The arguments for builder.

    Returns:
        tuple[str, dict[str, Any]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_builder(*all_arguments)


def multiprocess_exporter(
    id_exec: str,
    exps_export: Export,
    mode: str = "w+",
    indent: int = 2,
    encoding: str = "utf-8",
    jsonable: bool = False,
    mute: bool = True,
    pbar: Optional[tqdm.tqdm] = None,
) -> tuple[str, dict[str, Any]]:
    """Multiprocess exporter and writer for experiment.

    Args:
        id_exec (Hashable): ID of experiment.
        exps_export (Export): The export of experiment.
        mode (str, optional): The mode of writing. Defaults to "w+".
        indent (int, optional): The indent of writing. Defaults to 2.
        encoding (str, optional): The encoding of writing. Defaults to "utf-8".
        jsonable (bool, optional): The jsonable of writing. Defaults to False.
        mute (bool, optional): The mute of writing. Defaults to True.
        pbar (Optional[tqdm.tqdm], optional): The progress bar. Defaults to None.

    Returns:
        tuple[Hashable, dict[str, Any]]: The ID of experiment and the files of experiment.
    """
    qurryinfo_exp_id, qurryinfo_files = exps_export.write(
        mode=mode,
        indent=indent,
        encoding=encoding,
        jsonable=jsonable,
        mute=mute,
        multiprocess=False,
        pbar=pbar,
    )
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )

    return qurryinfo_exp_id, qurryinfo_files


def multiprocess_exporter_wrapper(
    all_arguments: tuple[str, Export, str, int, str, bool, bool, Optional[tqdm.tqdm]],
) -> tuple[str, dict[str, str]]:
    """Multiprocess wrapper for exporter.

    Args:
        all_arguments (tuple[str, Export, str, int, str, bool, bool, Optional[tqdm.tqdm]]):
            The arguments for exporter.

    Returns:
        tuple[str, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_exporter(*all_arguments)


def single_process_exporter(
    id_exec: str,
    exps_export: Export,
    mode: str = "w+",
    indent: int = 2,
    encoding: str = "utf-8",
    jsonable: bool = False,
    mute: bool = True,
    pbar: Optional[tqdm.tqdm] = None,
) -> tuple[str, dict[str, str]]:
    """Single process exporter and writer for experiment.

    Args:
        id_exec (Hashable): ID of experiment.
        exps_export (Export): The export of experiment.
        mode (str, optional): The mode of writing. Defaults to "w+".
        indent (int, optional): The indent of writing. Defaults to 2.
        encoding (str, optional): The encoding of writing. Defaults to "utf-8".
        jsonable (bool, optional): The jsonable of writing. Defaults to False.
        mute (bool, optional): The mute of writing. Defaults to True.
        pbar (Optional[tqdm.tqdm], optional): The progress bar. Defaults to None.

    Returns:
        tuple[Hashable, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    qurryinfo_exp_id, qurryinfo_files = exps_export.write(
        mode=mode,
        indent=indent,
        encoding=encoding,
        jsonable=jsonable,
        mute=mute,
        multiprocess=True,
        pbar=pbar,
    )
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )
    del exps_export
    gc.collect()
    return qurryinfo_exp_id, qurryinfo_files


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


def very_easy_chunk_distribution(
    respect_memory_array: list[tuple[str, int]],
    chunk_size: int = DEFAULT_POOL_SIZE,
) -> list[tuple[str, int]]:
    """Distribute the chunk for multiprocess.
    The chunk distribution is based on the number of CPU cores.

    Args:
        respect_memory_array (list[tuple[str, int]]):
            The array of respect memory.
            Each element is a tuple of (id, memory).
            The id is the ID of the experiment, and the memory is the memory usage.
            The array is sorted by the memory usage.
        chunk_size (int, optional):
            The chunk size. Defaults to DEFAULT_POOL_SIZE.

    Returns:
        list[tuple[str, int]]:
            The chunk distribution is a list of tuples of (id, memory).
    """

    ideal_chunks_num = int(np.ceil(len(respect_memory_array) / chunk_size))
    chunks_sorted_list = []
    for i in range(ideal_chunks_num):
        tmp = [
            respect_memory_array[i + j * ideal_chunks_num]
            for j in range(chunk_size)
            if i + j * ideal_chunks_num < len(respect_memory_array)
        ]
        chunks_sorted_list += tmp
    return chunks_sorted_list
