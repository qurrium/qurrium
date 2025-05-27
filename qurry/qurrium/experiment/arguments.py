"""The Arguments of Experiment (:mod:`qurry.qurrium.experiment.arguments`)"""

import json
from typing import Union, Optional, NamedTuple, TypedDict, Any, TypeVar
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from qiskit.providers import Backend

from ...declare import BaseRunArgs, TranspileArgs
from ...tools.backend import backend_name_getter
from ...tools.datetime import current_time, DatetimeDict
from ...capsule import jsonablize, DEFAULT_ENCODING

REQUIRED_FOLDER = ["args", "advent", "legacy", "tales", "reports"]
"""The required folder for exporting experiment."""

V5_TO_V7_FIELD = {
    "expName": "exp_name",
    "expID": "exp_id",
    "waveKey": "wave_key",
    "runArgs": "run_args",
    "transpileArgs": "transpile_args",
    "defaultAnalysis": "default_analysis",
    "saveLocation": "save_location",
    "summonerID": "summoner_id",
    "summonerName": "summoner_name",
}


def v5_to_v7_field_transpose(data_args: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The field name of v5 to v7.

    Args:
        data_args (dict[str, dict[str, Any]]): The arguments of experiment.

    Returns:
        dict[str, dict[str, Any]]: The arguments of experiment with new field name.
    """
    for k, nk in V5_TO_V7_FIELD.items():
        if k in data_args["commonparams"]:
            data_args["commonparams"][nk] = data_args["commonparams"].pop(k)
        if k in data_args["arguments"]:
            data_args["arguments"][nk] = data_args["arguments"].pop(k)
    return data_args


def wave_key_to_target_keys(wave_key: str) -> list[str]:
    """Convert the wave key to target keys.

    Args:
        wave_key (str): The wave key.

    Returns:
        list[str]: The target keys.
    """
    return [wave_key]


def v7_to_v9_field_transpose(data_args: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The field name of v7 to v9.

    Args:
        data_args (dict[str, dict[str, Any]]): The arguments of experiment.

    Returns:
        dict[str, dict[str, Any]]: The arguments of experiment with new field name
    """
    if "wave_key" in data_args["commonparams"]:
        data_args["commonparams"]["target_keys"] = wave_key_to_target_keys(
            data_args["commonparams"].pop("wave_key")
        )

    return data_args


@dataclass(frozen=True)
class ArgumentsPrototype:
    """Construct the experiment's parameters for specific options,
    which is overwritable by the inherition class."""

    exp_name: str
    """Name of experiment."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(self.__dict__.keys())

    def _asdict(self) -> dict[str, Any]:
        """The arguments as dictionary."""
        return self.__dict__

    @classmethod
    def _dataclass_fields(cls) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def _make(cls, iterable: Iterable):
        """Make the arguments."""
        return cls(*iterable)

    @classmethod
    def _filter(cls, *args, **kwargs):
        """Filter the arguments of the experiment.

        Args:
            *args: The arguments of the experiment.
            **kwargs: The keyword arguments of the experiment.

        Returns:
            tuple[ArgumentsPrototype, Commonparams, dict[str, Any]]:
                The arguments of the experiment,
                the common parameters of the experiment,
                and the side product of the experiment.
        """
        if len(args) > 0:
            raise ValueError("args filter can't be initialized with positional arguments.")
        infields = {}
        commonsinput = {}
        outfields = {}
        for k, v in kwargs.items():
            # pylint: disable=protected-access
            if k in cls._dataclass_fields():
                # pylint: enable=protected-access
                infields[k] = v
            elif k in Commonparams._fields:
                commonsinput[k] = v
            else:
                outfields[k] = v

        return (cls(**infields), Commonparams(**commonsinput), outfields)  # type: ignore


_A = TypeVar("_A", bound=ArgumentsPrototype)
"""Type variable for :cls:`ArgumentsPrototype`."""


class CommonparamsDict(TypedDict):
    """The export dictionary of :cls:`Commonparams`."""

    exp_name: str
    exp_id: str
    target_keys: list[Hashable]
    shots: int
    backend: Union[Backend, str]
    run_args: Union[BaseRunArgs, dict[str, Any]]
    transpile_args: TranspileArgs
    tags: tuple[str, ...]
    save_location: Union[Path, str]
    filename: str
    files: dict[str, Path]
    serial: Optional[int]
    summoner_id: Optional[str]
    summoner_name: Optional[str]
    datetimes: DatetimeDict


class Commonparams(NamedTuple):
    """Construct the experiment's parameters for system running."""

    exp_id: str
    """ID of experiment."""
    target_keys: list[Hashable]
    """The target keys of experiment."""

    # Qiskit argument of experiment.
    # Multiple jobs shared
    shots: int
    """Number of shots to run the program (default: 1024)."""
    backend: Union[Backend, str]
    """Backend to execute the circuits on, or the backend used."""
    run_args: Union[BaseRunArgs, dict[str, Any]]
    """Arguments of `execute`."""

    # Single job dedicated
    transpile_args: TranspileArgs
    """Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`."""

    tags: tuple[str, ...]
    """Tags of experiment."""

    # Arguments for exportation
    save_location: Union[Path, str]
    """Location of saving experiment. 
    If this experiment is called by :cls:`QurryMultiManager`,
    then `adventure`, `legacy`, `tales`, and `reports` will be exported 
    to their dedicated folders in this location respectively.
    This location is the default location for it's not specific 
    where to save when call :meth:`.write()`, if does, then will be overwriten and update."""
    filename: str
    """The name of file to be exported, 
    it will be decided by the :meth:`.export` when it's called.
    More info in the pydoc of :prop:`files` or :meth:`.export`.
    """
    files: dict[str, Path]
    """The list of file to be exported.

    ### Single experiment:

    For the :meth:`.write` function actually exports 4 different files
    respecting to `adventure`, `legacy`, `tales`, and `reports` like:

    .. code-block:: python
        files = {
            'folder': './bla_exp/',
            'qurryinfo': './bla_exp/qurryinfo.json',

            'args': './bla_exp/args/bla_exp.id={exp_id}.args.json',
            'advent': './bla_exp/advent/bla_exp.id={exp_id}.advent.json',
            'legacy': './bla_exp/legacy/bla_exp.id={exp_id}.legacy.json',
            'tales.dummyx1': './bla_exp/tales/bla_exp.id={exp_id}.dummyx1.json',
            'tales.dummyx2': './bla_exp/tales/bla_exp.id={exp_id}.dummyx2.json',
            ...
            'tales.dummyxn': './bla_exp/tales/bla_exp.id={exp_id}.dummyxn.json',
            'reports': './bla_exp/reports/bla_exp.id={exp_id}.reports.json',
            'reports.tales.dummyz1': './bla_exp/tales/bla_exp.id={exp_id}.dummyz1.reports.json',
            'reports.tales.dummyz2': './bla_exp/tales/bla_exp.id={exp_id}.dummyz2.reports.json',
            ...
            'reports.tales.dummyzm': './bla_exp/tales/bla_exp.id={exp_id}.dummyzm.reports.json',
        }

    which `bla_exp` is the example filename.

    ### Multi-experiment:

    If this experiment is called by :cls:`MultiManager`, 
    then the it will be named after `summoner_name` as known as the name of :cls:`MultiManager`.

    .. code-block:: python
        files = {
            'folder': './BLABLA_project/',
            'qurryinfo': './BLABLA_project/qurryinfo.json',

            'args': './BLABLA_project/args/index={serial}.id={exp_id}.args.json',
            'advent': './BLABLA_project/advent/index={serial}.id={exp_id}.advent.json',
            'legacy': './BLABLA_project/legacy/index={serial}.id={exp_id}.legacy.json',
            'tales.dummyx1': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyx1.json',
            'tales.dummyx2': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyx2.json',
            ...
            'tales.dummyxn': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyxn.json',
            'reports': './BLABLA_project/reports/index={serial}.id={exp_id}.reports.json',
            'reports.tales.dummyz1': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyz1.reports.json',
            'reports.tales.dummyz2': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyz2.reports.json',
            ...
            'reports.tales.dummyzm': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyzm.reports.json',
        }

    which `BLBLA_project` is the example :cls:`MultiManager` name 
    stored at :prop:`commonparams.summoner_name`.
    At this senerio, the :prop:`exp_name` will never apply as filename.

    """

    # Arguments for multi-experiment
    serial: Optional[int]
    """Index of experiment in a multiOutput."""
    summoner_id: Optional[str]
    """ID of experiment of :cls:`MultiManager`."""
    summoner_name: Optional[str]
    """Name of experiment of :cls:`MultiManager`."""

    # header
    datetimes: DatetimeDict
    """The datetime of experiment."""

    @staticmethod
    def default_value() -> CommonparamsDict:
        """The default value of each field."""
        return {
            "exp_name": "exps",
            "exp_id": "",
            "target_keys": [],
            "shots": -1,
            "backend": "",
            "run_args": {},
            "transpile_args": {},
            "tags": (),
            "save_location": Path("."),
            "filename": "unknown",
            "files": {},
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
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Read the exported experiment file.

        Args:
            exp_id (str): The ID of experiment.
            file_index (dict[str, str]): The index of exported experiment file.
            save_location (Path): The location of exported experiment file.

        Returns:
            tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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

        return (
            data_args["arguments"],
            data_args["commonparams"],
            data_args["outfields"],
        )

    def export(self) -> CommonparamsDict:
        """Export the experiment's common parameters.

        Returns:
            CommonparamsDict: The common parameters of experiment.
        """
        # pylint: disable=no-member
        commons: CommonparamsDict = jsonablize(self._asdict())
        # pylint: enable=no-member
        commons["backend"] = (
            self.backend if isinstance(self.backend, str) else backend_name_getter(self.backend)
        )
        return commons


def commons_dealing(
    commons_dict: dict[str, Any],
) -> dict[str, Any]:
    """Dealing some special commons arguments.

    Args:
        commons_dict (dict[str, Any]): The common parameters of the experiment.

    Returns:
        dict[str, Any]: The dealt common parameters of the experiment.
    """
    if "datetimes" not in commons_dict:
        commons_dict["datetimes"] = DatetimeDict({"bulid": current_time()})
    else:
        commons_dict["datetimes"] = DatetimeDict(commons_dict["datetimes"])
    if "tags" in commons_dict:
        if isinstance(commons_dict["tags"], list):
            commons_dict["tags"] = tuple(commons_dict["tags"])

    return commons_dict


def create_exp_args(
    arguments: Union[_A, dict[str, Any]],
    arguments_instance: type[_A],
) -> tuple[_A, dict[str, Any]]:
    """Create experiment arguments from the given arguments.

    Args:
        arguments (Union[_A, dict[str, Any]]): The arguments to be parsed.
        arguments_instance (type[_A]): The instance of the arguments class.
    Returns:
        tuple[_A, dict[str, Any]]: A tuple containing the parsed arguments instance and
            a dictionary of deprecated fields.
    Raises:
        TypeError: If the arguments is not an instance of the arguments class or a dictionary.
    """

    if isinstance(arguments, arguments_instance):
        return arguments, {}

    if isinstance(arguments, dict):
        arguments_deprecated = {}
        arg_parsed = {}
        # pylint: disable=protected-access
        dataclass_fields = arguments_instance._dataclass_fields()
        for k, v in arguments.items():
            if k in dataclass_fields:
                arg_parsed[k] = v
            else:
                arguments_deprecated[k] = v
        # pylint: enable=protected-access
        return arguments_instance(**arg_parsed), arguments_deprecated

    raise TypeError(f"arguments should be {arguments_instance} or dict, not {type(arguments)}")


def create_exp_commons(
    commons: Union[Commonparams, dict[str, Any]],
) -> tuple[Commonparams, dict[str, Any]]:
    """Create experiment commons from the given commons.

    Args:
        commons (Union[Commonparams, dict[str, Any]]): The commons to be parsed.
    Returns:
        tuple[Commonparams, dict[str, Any]]: A tuple containing the parsed commons instance and
            a dictionary of deprecated fields.
    Raises:
        TypeError: If the commons is not an instance of the commons class or a dictionary.
    """

    if isinstance(commons, Commonparams):
        return commons, {}

    if isinstance(commons, dict):
        commons_deprecated = {}
        commons_parsed = {}
        for k, v in commons.items():
            if k in Commonparams._fields:
                commons_parsed[k] = v
            else:
                commons_deprecated[k] = v
        return Commonparams(**commons_dealing(commons_parsed)), commons_deprecated

    raise TypeError(f"commons should be {Commonparams} or dict, not {type(commons)}")


def create_exp_outfields(
    outfields: Union[dict[str, Any], None],
) -> dict[str, Any]:
    """Create experiment outfields from the given outfields.

    Args:
        outfields (Union[dict[str, Any], None]): The outfields to be parsed.
    Returns:
        dict[str, Any]: The parsed outfields.
    Raises:
        TypeError: If the outfields is not a dictionary or None.
    """

    if outfields is None:
        return {}

    if isinstance(outfields, dict):
        return outfields

    raise TypeError(f"outfields should be dict or None, not {type(outfields)}")
