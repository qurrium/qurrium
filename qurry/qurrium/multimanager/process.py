"""Multi-process component for multimanager (:mod:`qurry.qurrium.multimanager.process`)"""

from typing import Any
from pathlib import Path

from .container import _E
from ..experiment import ExperimentPrototype, Export


def multiprocess_exporter(
    id_exec: str,
    exps_export: Export,
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
    qurryinfo_exp_id, qurryinfo_files = exps_export.write()
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )
    del exps_export

    return qurryinfo_exp_id, qurryinfo_files


def multiprocess_exporter_wrapper(
    all_arguments: tuple[str, Export],
) -> tuple[str, dict[str, str]]:
    """Multiprocess wrapper for exporter.

    Args:
        all_arguments (tuple[str, Export, str, int, str, bool]):
            The arguments for exporter.

            - id_exec (str): ID of experiment.
            - exps_export (Export): The export of experiment.

    Returns:
        tuple[str, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_exporter(*all_arguments)


def multiprocess_writer(
    id_exec: str,
    exps: ExperimentPrototype,
    save_location: Path,
    export_transpiled_circuit: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Multiprocess exporter and writer for experiment.

    Args:
        id_exec (Hashable): ID of experiment.
        exps (ExperimentPrototype): The export of experiment.
        save_location (Path): The location of saving.
        export_transpiled_circuit (bool, optional):
            Whether to export transpiled circuit. Defaults to False.

    Returns:
        tuple[Hashable, dict[str, Any]]: The ID of experiment and the files of experiment.
    """
    export_instance = exps.export(
        save_location=save_location,
        export_transpiled_circuit=export_transpiled_circuit,
    )
    qurryinfo_exp_id, qurryinfo_files = export_instance.write()
    assert id_exec == qurryinfo_exp_id, (
        f"{id_exec} is not equal to {qurryinfo_exp_id}" + " which is not supported."
    )
    del export_instance

    return qurryinfo_exp_id, qurryinfo_files


def multiprocess_writer_wrapper(
    all_arguments: tuple[str, _E, Path, bool],
) -> tuple[str, dict[str, str]]:
    """Multiprocess wrapper for exporter.

    Args:
        all_arguments (tuple[str, ExperimentPrototype, Path, bool]):
            The arguments for exporter.

            - id_exec (str): ID of experiment.
            - exps (ExperimentPrototype): The export of experiment.
            - save_location (Path): The location of saving.
            - export_transpiled_circuit (bool): Whether to export transpiled circuit.

    Returns:
        tuple[str, dict[str, str]]: The ID of experiment and the files of experiment.
    """
    return multiprocess_writer(*all_arguments)
