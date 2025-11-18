"""The Common Parameters (:mod:`qurry.qurrium.arguments.commonparams`)"""

import json
from typing import Union, Optional, NamedTuple, TypedDict, Any
from pathlib import Path

from qiskit.providers import Backend

from .utils import (
    v5_to_v7_field_transpose,
    v7_to_v9_field_transpose,
    raw_commons_process,
    filter_deprecated_args,
)
from ..container import BaseRunArgs, TranspileArgs, WCKeyable
from ...tools.backend import backend_name_getter
from ...tools.datetime import DatetimeDict
from ...capsule import jsonablize, DEFAULT_ENCODING


class CommonparamsDict(TypedDict):
    """The export dictionary of :class:`Commonparams`."""

    exp_id: str
    """ID of experiment."""
    target_keys: list[WCKeyable]
    """The target keys of experiment, 
    which is the list of key of the used waves in this experiment."""

    shots: int
    """Number of shots to run the program."""
    backend: Union[Backend, str]
    """Backend to execute the circuits on, or the backend used."""
    run_args: Union[BaseRunArgs, dict[str, Any]]
    """Arguments for :meth:`~qiskit.providers.backend.BackendV2.run`"""
    transpile_args: TranspileArgs
    """Arguments of :func:`~qiskit.compiler.transpile`."""

    tags: tuple[str, ...]
    """Tags of experiment."""

    save_location: Union[Path, str]
    """Location of saving experiment."""

    serial: Optional[int]
    """Index of experiment in :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""
    summoner_id: Optional[str]
    """ID of experiment of :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""
    summoner_name: Optional[str]
    """Name of experiment of :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""

    datetimes: DatetimeDict
    """The datetime of experiment."""


class CommonparamsReadReturn(TypedDict):
    """The return type of :meth:`Commonparams.read_with_arguments`.

    This includes the experiment's arguments,
    the experiment's common parameters, and the experiment's side product.

    Attention, those result are unprocessed, so we define them as `dict[str, Any]`.
    """

    arguments: dict[str, Any]
    commonparams: dict[str, Any]
    outfields: dict[str, Any]


class Commonparams(NamedTuple):
    """Construct the experiment's parameters for running."""

    exp_id: str
    """ID of experiment."""
    target_keys: list[WCKeyable]
    """The target keys of experiment, 
    which is the list of key of the used waves in this experiment."""

    # Qiskit argument of experiment.
    # Multiple jobs shared
    shots: int
    """Number of shots to run the program."""
    backend: Union[Backend, str]
    """Backend to execute the circuits on, or the backend used."""
    run_args: Union[BaseRunArgs, dict[str, Any]]
    """Arguments for :meth:`~qiskit.providers.backend.BackendV2.run`"""

    # Single job dedicated
    transpile_args: TranspileArgs
    """Arguments of :func:`~qiskit.compiler.transpile`."""

    tags: tuple[str, ...]
    """Tags of experiment."""

    # Arguments for exportation
    save_location: Union[Path, str]
    """Location of saving experiment. 
    If this experiment is called by
    :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`,
    then `adventure`, `legacy`, `tales`, and `reports` will be exported 
    to their dedicated folders in this location respectively.
    This location is the default location for it's not specific 
    where to save when call 
    :meth:`~qurry.qurrium.experiment.experiment.ExperimentPrototype.write`, 
    if does, then will be overwriten and update."""

    # Arguments for multi-experiment
    serial: Optional[int]
    """Index of experiment in :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""
    summoner_id: Optional[str]
    """ID of experiment of :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""
    summoner_name: Optional[str]
    """Name of experiment of :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""

    # header
    datetimes: DatetimeDict
    """The datetime of experiment."""

    @staticmethod
    def default_value() -> CommonparamsDict:
        """The default value of each field."""
        return {
            "exp_id": "",
            "target_keys": [],
            "shots": -1,
            "backend": "",
            "run_args": {},
            "transpile_args": {},
            "tags": (),
            "save_location": Path("."),
            "serial": None,
            "summoner_id": None,
            "summoner_name": None,
            "datetimes": DatetimeDict(),
        }

    @classmethod
    def read_with_arguments(
        cls,
        exp_id: str,
        file_index: dict[str, str],
        save_location: Path,
    ) -> CommonparamsReadReturn:
        """Read the exported experiment file.

        This includes the experiment's arguments,
        the experiment's common parameters, and the experiment's side product.

        Attention, those result are unprocessed, so we define them as `dict[str, Any]`.

        Args:
            exp_id (str): The ID of experiment.
            file_index (dict[str, str]): The index of exported experiment file.
            save_location (Path): The location of exported experiment file.

        Returns:
            CommonparamsReadReturn
                The experiment's arguments,
                the experiment's common parameters,
                and the experiment's side product.
        """
        raw_data = {}
        with open(save_location / file_index["args"], "r", encoding=DEFAULT_ENCODING) as f:
            raw_data = json.load(f)
        data_args: dict[str, dict[str, Any]] = {
            "arguments": raw_data["arguments"],
            "commonparams": raw_data["commonparams"],
            "outfields": raw_data["outfields"],
        }

        data_args = v5_to_v7_field_transpose(data_args)
        data_args = v7_to_v9_field_transpose(data_args)

        assert data_args["commonparams"]["exp_id"] == exp_id, "The exp_id is not match."

        return {
            "arguments": data_args["arguments"],
            "commonparams": data_args["commonparams"],
            "outfields": data_args["outfields"],
        }

    def export(self) -> CommonparamsDict:
        """Export the experiment's common parameters.

        Returns:
            CommonparamsDict: The common parameters of experiment.
        """
        # pylint: disable=no-member
        commons: CommonparamsDict = jsonablize(self._asdict())
        # pylint: enable=no-member
        commons["backend"] = backend_name_getter(self.backend)
        return commons

    @classmethod
    def create(
        cls, commons: Union["Commonparams", dict[str, Any]]
    ) -> tuple["Commonparams", dict[str, Any]]:
        """Create experiment commons from the given commons.

        Args:
            commons (Union[Commonparams, dict[str, Any]]): The commons to be parsed.

        Raises:
            TypeError: If the commons is not an instance of the commons class or a dictionary.

        Returns:
            A tuple containing the parsed commons instance and a dictionary of deprecated fields.
        """

        if isinstance(commons, cls):
            return commons, {}
        if isinstance(commons, dict):
            commons_parsed, commons_deprecated = filter_deprecated_args(commons, cls._fields)
            return cls(**raw_commons_process(commons_parsed)), commons_deprecated

        raise TypeError(f"commons should be {cls} or dict, not {type(commons)}")
